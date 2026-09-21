"""API conversation, tool evidence and failure boundaries (no network)."""
import copy
import io
import json
import sys

from PIL import Image
import pytest

from cloth_agent import responses_remote_runner as runner
from cloth_agent import image_tools_mcp, codex_remote_runner
from cloth_agent.planner_backend import RemoteCodexBackend, RemoteClaudeBackend, PlannerBackendError


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'image_tools', image_tools_mcp)
    monkeypatch.setitem(sys.modules, 'codex_adapter', codex_remote_runner)
    monkeypatch.setattr(runner, 'load_provider', lambda _: ('https://test.invalid/responses', {}, 'test-secret'))
    Image.new('RGB', (24, 16), 'red').save(tmp_path / 'image_0.png')
    request = {'profile': 'rbs', 'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
               'max_output_tokens': 32768, 'max_tool_calls': 3, 'image_count': 1,
               'image_edit_limit': 1, 'system_prompt': 'Inspect RGB',
               'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                          'required': ['ok'], 'additionalProperties': False}}
    return tmp_path, request


def completed(index, output, **extras):
    return {'id': f'resp_{index}', 'status': 'completed', 'output': output, **extras}


def call(identity='call_1', tool='view_image', **arguments):
    return {'type': 'function_call', 'call_id': identity, 'name': tool,
            'arguments': json.dumps(arguments or {'image_id': 'image_0'})}


FINAL = {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ok":true}'}]}


def test_multi_tool_conversation_preserves_reasoning_images_and_budget(job, capsys):
    directory, request = job
    seen = []
    reasoning = {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'}

    def api(url, headers, payload, timeout):
        seen.append(copy.deepcopy(payload))
        assert payload['max_output_tokens'] == 32768
        assert payload['store'] is False
        assert payload['instructions'].startswith('Inspect RGB')
        assert payload['text']['format']['strict'] is True
        assert 0 < timeout <= 300
        if len(seen) == 1:
            return completed(1, [reasoning, call()])
        history = payload['input']
        assert reasoning in history
        assert any(x.get('type') == 'function_call' for x in history)
        result = history[-1]['output']
        if len(seen) == 2:
            assert result[-1]['type'] == 'input_image'
            assert result[-1]['image_url'].startswith('data:image/png;base64,')
            return completed(2, [call('call_2', 'map_point', image_id='image_0', pixel_xy=[3, 4])])
        assert all(x['type'] == 'input_text' for x in result)
        assert json.loads(result[0]['text'])['pixel_xy'] == [3, 4]
        return completed(3, [FINAL])

    runner.run(directory, request, 'Inspect and map', api)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['subtype'] == 'success'
    assert events[-1]['provider'] == 'responses'
    assert events[-1]['num_tool_calls'] == 2
    assert events[-1]['response_count'] == 3
    assert events[-1]['structured_output'] == {'ok': True}
    assert len([e for e in events if e.get('type') == 'user']) == 2


@pytest.mark.parametrize('status', ['incomplete', 'failed', 'in_progress'])
def test_unsuccessful_response_never_delivers_pending_image_or_success(job, capsys, status):
    directory, request = job
    attempts = []

    def api(*args):
        attempts.append(1)
        if len(attempts) == 1:
            return completed(1, [call()])
        return {'status': status, 'output': [FINAL],
                'incomplete_details': {'reason': 'max_output_tokens'}}

    with pytest.raises(RuntimeError, match='max_output_tokens'):
        runner.run(directory, request, '', api)
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert not any(e.get('type') in {'user', 'result'} for e in events)
    assert len(attempts) == 2


@pytest.mark.parametrize('bad_call,match', [
    (call(tool='robot_move'), 'unexpected'),
    (call(identity=''), 'missing/duplicate'),
])
def test_only_registered_tools_with_unique_ids_execute(job, bad_call, match):
    directory, request = job
    with pytest.raises(ValueError, match=match):
        runner.run(directory, request, '', lambda *args: completed(1, [bad_call]))
    assert not (directory / 'image_tool_calls.jsonl').exists()


def test_budget_checked_before_tool_execution(job):
    directory, request = job
    request['max_tool_calls'] = 1
    with pytest.raises(ValueError, match='budget'):
        runner.run(directory, request, '', lambda *args: completed(1, [call(), call('call_2')]))
    assert not (directory / 'image_tool_calls.jsonl').exists()


def test_edit_budget_and_tool_errors_remain_in_conversation(job, capsys):
    directory, request = job
    request['image_edit_limit'] = 0
    attempts = []

    def api(url, headers, payload, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            return completed(1, [call(tool='rotate_image', image_id='image_0', degrees_clockwise=90)])
        if len(attempts) == 2:
            assert 'budget' in payload['input'][-1]['output'][0]['text'].lower()
            return completed(2, [call('call_2')])
        return completed(3, [FINAL])

    runner.run(directory, request, '', api)
    assert not list(directory.glob('view_*.png'))
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    errors = [e for e in events if e.get('type') == 'user' and e['message']['content'][0]['is_error']]
    assert len(errors) == 1


def test_provider_uses_selected_profile_and_named_environment_only(tmp_path, monkeypatch):
    (tmp_path / 'config.toml').write_text('[model_providers.company]\nwire_api="responses"\n'
                                        'base_url="https://company.invalid/v1"\nenv_key="TEST_COMPANY_KEY"\n')
    (tmp_path / 'rbs.config.toml').write_text('model_provider="company"\n')
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    monkeypatch.setenv('TEST_COMPANY_KEY', 'local-test-only')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://wrong.invalid/v1')
    url, headers, key = runner.load_provider('rbs')
    assert url == 'https://company.invalid/v1/responses'
    assert headers['Authorization'] == 'Bearer local-test-only'
    monkeypatch.delenv('TEST_COMPANY_KEY')
    with pytest.raises(ValueError, match='credential'):
        runner.load_provider('rbs')


def test_api_error_does_not_echo_secret(monkeypatch):
    import urllib.error
    class Failing:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 401, 'bad', {},
                                         io.BytesIO(b'key: test-secret sk-abc***xyz'))
    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: Failing())
    with pytest.raises(RuntimeError) as caught:
        runner.post('https://test.invalid/responses', {'Authorization': 'Bearer test-secret'}, {}, 1)
    assert 'test-secret' not in str(caught.value)
    assert 'sk-' not in str(caught.value)


def test_responses_backend_never_restarts_token_failed_job(monkeypatch):
    attempts = []
    def invoke(*a, **kw):
        attempts.append(1)
        raise PlannerBackendError('max_output_tokens')
    monkeypatch.setattr(RemoteClaudeBackend, '_invoke', invoke)
    with pytest.raises(PlannerBackendError):
        RemoteCodexBackend()._invoke(schema={})
    assert len(attempts) == 1


def test_provider_budget_violation_is_reported(job, capsys):
    directory, request = job
    request['max_output_tokens'] = 16
    responses = iter([completed(1, [call()]), completed(2, [FINAL], usage={'output_tokens': 314})])
    runner.run(directory, request, '', lambda *args: next(responses))
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    warnings = [e for e in events if e.get('subtype') == 'responses_budget_warning']
    assert warnings[0]['reported_output_tokens'] == 314
    assert warnings[0]['requested_max_output_tokens'] == 16


def test_metadata_only_cannot_satisfy_visual_evidence(job):
    directory, request = job
    responses = iter([completed(1, [call(tool='image_info')]), completed(2, [FINAL])])
    with pytest.raises(ValueError, match='no RGB tool evidence'):
        runner.run(directory, request, '', lambda *args: next(responses))


def test_transport_error_preserves_diagnostic_and_never_delivers_pending_image(job, capsys, monkeypatch):
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    directory, request = job
    attempts = []
    diagnostic = {'category': 'HTTP_STATUS', 'http_status': 524,
                  'phase': 'reading_http_error', 'elapsed_s': 120,
                  'client_request_id': 'test-client-id', 'detail': 'HTTP 524'}

    def api(*args):
        attempts.append(1)
        if len(attempts) == 1:
            return completed(1, [call()])
        raise runner.ResponsesRequestError(diagnostic)

    with pytest.raises(runner.ResponsesRequestError) as caught:
        runner.run(directory, request, '', api)
    assert caught.value.diagnostic['http_status'] == 524
    assert caught.value.diagnostic['retry_decision'] == 'attempts_exhausted'
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert not any(event.get('type') in {'user', 'result'} for event in events)
    assert len(attempts) == 4  # One successful tool request, three failed result submissions.


def failure(category='STREAM_EOF', **fields):
    return runner.ResponsesRequestError({'category': category, 'phase': 'reading_sse',
        'elapsed_s': 21.58, 'client_request_id': 'wire-id', 'http_status': 200,
        'detail': 'simulated transport failure', **fields})


@pytest.mark.parametrize('trial', range(10))
def test_retry_same_image_result_without_repeating_tools_or_resetting_budget(job, capsys, monkeypatch, trial):
    directory, request = job
    request['max_tool_calls'] = 1
    request['image_edit_limit'] = 1
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    seen, ids = [], []

    def api(url, headers, payload, timeout):
        seen.append(copy.deepcopy(payload))
        ids.append(headers['X-Client-Request-Id'])
        if len(seen) == 1:
            return completed(1, [call(tool='rotate_image', image_id='image_0', degrees_clockwise=90)])
        # No read-hook/delivery evidence may be emitted until a retry succeeds.
        audit = [json.loads(line) for line in (directory / 'image_tool_calls.jsonl').read_text().splitlines()]
        assert not any(e.get('kind') == 'tool_lifecycle' for e in audit)
        if len(seen) == 2:
            raise failure()
        if len(seen) == 3:
            raise failure('HTTP_STATUS', http_status=524)
        return completed(2, [FINAL])

    runner.run(directory, request, f'trial {trial}', api)
    assert seen[1] == seen[2] == seen[3]
    assert len(set(ids)) == 4
    assert seen[-1]['reasoning']['effort'] == 'medium'
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    final = events[-1]
    assert final['http_attempt_count'] == 4
    assert final['request_retry_count'] == 2
    assert final['response_count'] == 2
    assert final['num_tool_calls'] == 1
    assert len([e for e in events if e.get('type') == 'user']) == 1
    audit = [json.loads(line) for line in (directory / 'image_tool_calls.jsonl').read_text().splitlines()]
    edits = [e for e in audit if e.get('tool') == 'rotate_image' and e.get('status') == 'ok']
    assert len(edits) == 1
    assert edits[0]['edit_budget']['used'] == 1
    retries = [e for e in events if e.get('event') == 'retry_scheduled']
    assert len(retries) == 2
    assert retries[0]['payload_sha256'] == retries[1]['payload_sha256']


@pytest.mark.parametrize('error', [
    failure('HTTP_STATUS', http_status=401), failure('HTTP_STATUS', http_status=400),
    failure('HTTP_STATUS', http_status=429), failure('HTTP_STATUS', http_status=501),
    failure('HTTP_STATUS', http_status=503, upstream_error={'type': 'invalid_request_error'}),
    failure('OUTPUT_TOKEN_LIMIT'), failure('INVALID_STREAM'), failure('INVALID_RESPONSE'),
    failure('FINAL_JSON_INVALID'), failure('TOOL_ARGUMENTS_INVALID'), failure('TLS_ERROR'),
    failure('DNS_ERROR'), failure('STREAM_API_ERROR'), failure('RESPONSE_NOT_COMPLETED'),
    failure('REQUEST_DEADLINE'), failure('NETWORK_ERROR', exception_type='PermissionError'),
])
def test_permanent_or_unknown_errors_are_not_retried(job, monkeypatch, error):
    directory, request = job
    attempts = []
    monkeypatch.setattr(runner.time, 'sleep', lambda _: pytest.fail('must not back off'))

    def api(*args):
        attempts.append(1)
        raise error

    with pytest.raises(runner.ResponsesRequestError) as caught:
        runner.run(directory, request, '', api)
    assert len(attempts) == 1
    assert caught.value.diagnostic['retry_decision'] == 'not_retryable'


@pytest.mark.parametrize('error', [failure('CLIENT_TIMEOUT'),
    failure('NETWORK_ERROR', exception_type='ConnectionResetError'),
    failure('NETWORK_ERROR', exception_type='IncompleteRead'),
    failure('STREAM_API_ERROR', upstream_error={'code': 'service_unavailable'}),
    failure('RESPONSE_NOT_COMPLETED', upstream_error={'type': 'service_unavailable_error'}),
    failure('HTTP_STATUS', http_status=503)])
def test_transient_failures_have_at_most_three_attempts(job, monkeypatch, error):
    directory, request = job
    calls = []
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)

    def api(*args):
        calls.append(1)
        raise error

    with pytest.raises(runner.ResponsesRequestError) as caught:
        runner.run(directory, request, '', api)
    assert len(calls) == 3
    assert caught.value.diagnostic['retry_decision'] == 'attempts_exhausted'


@pytest.mark.parametrize('remaining,header,decision', [
    (1, None, 'conversation_deadline'), (100, '61', 'retry_after_too_long'),
    (10, '10', 'conversation_deadline'), (100, 'inf', 'retry_after_too_long')])
def test_retry_wait_never_overruns_deadline_or_server_delay(job, monkeypatch, remaining, header, decision):
    directory, request = job
    request['timeout_s'] = 100
    clock = [0.0]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runner.time, 'sleep', lambda _: pytest.fail('no budget to retry'))
    calls = []

    def api(*args):
        calls.append(1)
        clock[0] = 100 - remaining
        raise failure('HTTP_STATUS', http_status=503, response_headers={'retry-after': header})

    with pytest.raises(runner.ResponsesRequestError) as caught:
        runner.run(directory, request, '', api)
    assert len(calls) == 1
    assert caught.value.diagnostic['retry_decision'] == decision


def test_retry_after_and_attempt_timeouts_share_one_deadline(job, monkeypatch):
    directory, request = job
    request['timeout_s'] = 50
    clock, timeouts, waits = [0.0], [], []
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runner.random, 'uniform', lambda *a: 0)

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(runner.time, 'sleep', sleep)

    def api(url, headers, payload, timeout):
        timeouts.append(timeout)
        clock[0] += 10
        if len(timeouts) == 1:
            raise failure('HTTP_STATUS', http_status=503, response_headers={'retry-after': '7'})
        if len(timeouts) == 2:
            return completed(1, [call()])
        return completed(2, [FINAL])

    runner.run(directory, request, '', api)
    assert waits == [7]
    assert timeouts == [50, 33, 23]


def test_retry_after_http_date(monkeypatch):
    monkeypatch.setattr(runner.time, 'time', lambda: 0)
    monkeypatch.setattr(runner.random, 'uniform', lambda *a: 0)
    assert runner.retry_delay({'response_headers': {'retry-after': 'Thu, 01 Jan 1970 00:00:09 GMT'}}, 1) == 9


def test_production_default_is_medium():
    assert RemoteCodexBackend.reasoning_effort == 'medium'


def test_nested_overload_still_yields_to_permanent_error_code():
    assert not runner.retryable_request_error({'category': 'STREAM_API_ERROR',
        'upstream_error': {'type': 'service_unavailable', 'code': 'insufficient_quota'}})
