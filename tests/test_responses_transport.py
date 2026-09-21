"""Responses wire failures, stream boundaries and diagnostic persistence."""
from email.message import Message
import io
import json
import socket
import ssl
import urllib.error

import pytest
from PIL import Image

from cloth_agent import responses_remote_runner as runner
from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.claude_stream import ClaudeStreamProgress


FINAL = {'id': 'resp_test', 'status': 'completed', 'output': [
    {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ok":true}'}]}]}


def sse(*events):
    return b''.join(b'data: ' + json.dumps(e).encode() + b'\r\n\r\n' for e in events)


class Response(io.BytesIO):
    code = 200

    def __init__(self, body, content_type='text/event-stream'):
        super().__init__(body)
        self.headers = Message()
        self.headers['content-type'] = content_type
        self.headers['x-request-id'] = 'gateway-request-123'
        self.headers['cf-ray'] = 'edge-123'
        self.headers['set-cookie'] = 'never-save-this'


def install(monkeypatch, response):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(request)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: Opener())
    return sent


def post():
    return runner.post('https://example.invalid/v1/responses',
                       {'Authorization': 'Bearer test-secret', 'X-Private': 'second-secret'},
                       {'stream': True, 'input': 'do-not-log-prompt'}, 2)


def diagnostics(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_stream_terminal_preserves_complete_response_and_logs_no_content(monkeypatch, capsys):
    body = b': keepalive\n\n' + sse(
        {'type': 'response.created', 'response': {'id': 'resp_test'}},
        {'type': 'response.output_text.delta', 'delta': 'do-not-log-delta'},
        {'type': 'response.completed', 'response': FINAL})
    sent = install(monkeypatch, Response(body))
    assert post() == FINAL
    assert sent[0].get_header('X-client-request-id')
    rows = diagnostics(capsys)
    last = rows[-1]
    assert last['event'] == 'completed'
    assert last['event_count'] == 3
    assert last['last_event'] == 'response.completed'
    assert last['first_event_s'] >= last['headers_received_s']
    assert last['response_headers']['x-request-id'] == 'gateway-request-123'
    assert last['response_format'] == 'sse'
    output = json.dumps(rows)
    for hidden in ('test-secret', 'second-secret', 'do-not-log', 'set-cookie', 'never-save-this'):
        assert hidden not in output


@pytest.mark.parametrize('body,category', [
    (sse({'type': 'response.output_text.delta', 'delta': '{"ok":true}'}), 'STREAM_EOF'),
    (b'data: [DONE]\n\n', 'STREAM_EOF'),
    (b'data: {bad}\n\n', 'INVALID_STREAM'),
    (sse({'type': 'response.completed', 'response': {**FINAL, 'status': 'in_progress'}}), 'INVALID_STREAM'),
    (sse({'type': 'error', 'code': 'bad_response_status_code',
          'message': 'upstream HTTP 524'}), 'STREAM_API_ERROR'),
    (sse({'type': 'response.incomplete', 'response': {**FINAL, 'status': 'incomplete',
          'incomplete_details': {'reason': 'max_output_tokens'}}}), 'OUTPUT_TOKEN_LIMIT'),
    (sse({'type': 'response.failed', 'response': {**FINAL, 'status': 'failed',
          'error': {'code': 'server_error', 'message': 'upstream failed'}}}), 'RESPONSE_NOT_COMPLETED'),
])
def test_partial_and_failed_streams_rejected(monkeypatch, capsys, body, category):
    install(monkeypatch, Response(body))
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    assert caught.value.diagnostic['category'] == category
    assert caught.value.diagnostic['http_status'] == 200
    assert diagnostics(capsys)[-1]['event'] == 'failed'


def test_gateway_ignoring_stream_is_explicit(monkeypatch, capsys):
    install(monkeypatch, Response(json.dumps(FINAL).encode(), 'application/json'))
    assert post() == FINAL
    last = diagnostics(capsys)[-1]
    assert last['stream_fallback'] is True
    assert last['first_event_s'] is None
    assert last['event_count'] == 0


def test_rbs_empty_terminal_uses_finished_items_only(monkeypatch, capsys):
    item = {'id': 'fc_1', 'type': 'function_call', 'call_id': 'call_1',
            'name': 'view_image', 'arguments': '{"image_id":"image_0"}', 'status': 'completed'}
    body = sse({'type': 'response.created', 'response': {'id': 'resp_test'}},
               {'type': 'response.output_item.added', 'output_index': 0,
                'item': {**item, 'arguments': '', 'status': 'in_progress'}},
               {'type': 'response.function_call_arguments.delta', 'delta': 'incorrect partial'},
               {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
               {'type': 'response.completed', 'response': {**FINAL, 'output': []}})
    install(monkeypatch, Response(body))
    result = post()
    assert result['output'] == [item]
    assert diagnostics(capsys)[-1]['output_source'] == 'output_item.done'


@pytest.mark.parametrize('index,status,terminal_id,extra', [
    (1, 'completed', 'resp_test', []),
    (0, 'in_progress', 'resp_test', []),
    (0, 'completed', 'wrong_response', []),
    (0, 'completed', 'resp_test', [{'type': 'response.output_item.added',
                                  'output_index': 1, 'item': {'id': 'unfinished'}}]),
])
def test_reconstruction_rejects_gaps_unfinished_items_and_wrong_identity(
        monkeypatch, index, status, terminal_id, extra):
    body = sse({'type': 'response.created', 'response': {'id': 'resp_test'}},
               {'type': 'response.output_item.done', 'output_index': index,
                'item': {'id': 'item_1', 'type': 'function_call', 'status': status}},
               *extra, {'type': 'response.completed',
                        'response': {**FINAL, 'id': terminal_id, 'output': []}})
    install(monkeypatch, Response(body))
    with pytest.raises(runner.ResponsesRequestError, match='INVALID_STREAM'):
        post()


def test_http_524_preserves_gateway_identifiers_and_redacts_errors(monkeypatch, capsys):
    response = Response(b'')
    error = urllib.error.HTTPError('https://example.invalid', 524, 'timeout', response.headers,
                                  io.BytesIO(b'{"error":{"code":"bad_response_status_code",'
                                             b'"message":"test-secret second-secret sk-private"}}'))
    install(monkeypatch, error)
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    result = caught.value.diagnostic
    assert result['category'] == 'HTTP_STATUS'
    assert result['http_status'] == 524
    assert result['response_headers']['cf-ray'] == 'edge-123'
    assert result['upstream_error']['code'] == 'bad_response_status_code'
    assert result['first_event_s'] is None
    output = json.dumps(diagnostics(capsys)) + str(caught.value)
    for hidden in ('test-secret', 'second-secret', 'sk-private'):
        assert hidden not in output


@pytest.mark.parametrize('error,category', [
    (urllib.error.URLError(socket.gaierror(-2, 'DNS failed')), 'DNS_ERROR'),
    (urllib.error.URLError(ssl.SSLError('TLS failed')), 'TLS_ERROR'),
    (TimeoutError('socket timed out'), 'CLIENT_TIMEOUT'),
    (ConnectionResetError('reset'), 'NETWORK_ERROR'),
])
def test_preheader_failure_is_not_misreported_as_http(monkeypatch, error, category):
    install(monkeypatch, error)
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    result = caught.value.diagnostic
    assert result['category'] == category
    assert result['phase'] == 'awaiting_headers'
    assert result['http_status'] is None


def test_timeout_after_stream_started_keeps_last_event(monkeypatch):
    class TimeoutBody(Response):
        def readline(self, size):
            if self.tell() == len(self.getvalue()):
                raise TimeoutError('read timeout')
            return super().readline(size)

    install(monkeypatch, TimeoutBody(sse({'type': 'response.created'})))
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    result = caught.value.diagnostic
    assert result['category'] == 'CLIENT_TIMEOUT'
    assert result['phase'] == 'reading_sse'
    assert result['last_event'] == 'response.created'
    assert result['http_status'] == 200


def test_http_error_body_timeout_does_not_hide_524(monkeypatch):
    class TimeoutBody(io.BytesIO):
        def read1(self, size):
            raise TimeoutError('read timeout')

    install(monkeypatch, urllib.error.HTTPError('https://example.invalid', 524, 'timeout',
                                              {}, TimeoutBody()))
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    assert caught.value.diagnostic['http_status'] == 524
    assert caught.value.diagnostic['category'] == 'HTTP_STATUS'


def test_local_validation_retains_http_diagnostics(monkeypatch, capsys):
    install(monkeypatch, Response(sse({'type': 'response.completed', 'response': FINAL})))
    result = post()
    with pytest.raises(runner.ResponsesRequestError) as caught:
        runner.validation_error(result, 'FINAL_JSON_INVALID', 'Missing final JSON')
    diagnostic = caught.value.diagnostic
    assert diagnostic['http_status'] == 200
    assert diagnostic['phase'] == 'validating_response'
    assert diagnostic['response_id'] == 'resp_test'
    assert diagnostic['response_headers']['x-request-id'] == 'gateway-request-123'
    assert diagnostics(capsys)[-1]['category'] == 'FINAL_JSON_INVALID'


def test_deadline_checked_even_when_body_keeps_arriving(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: now[0])

    class SlowBody(Response):
        def readline(self, size):
            now[0] += 3
            return b': keepalive\n'

    install(monkeypatch, SlowBody(b''))
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    assert caught.value.diagnostic['category'] == 'REQUEST_DEADLINE'


def test_failed_response_without_output_keeps_upstream_error(monkeypatch):
    install(monkeypatch, Response(sse({'type': 'response.failed', 'response': {
        'id': 'resp_test', 'status': 'failed',
        'error': {'code': 'bad_response_status_code', 'message': 'HTTP 524'}}})))
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    result = caught.value.diagnostic
    assert result['category'] == 'RESPONSE_NOT_COMPLETED'
    assert result['http_status'] == 200
    assert result['upstream_error']['code'] == 'bad_response_status_code'


def test_diagnostics_saved_and_shown_in_live_progress(tmp_path):
    image = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(image)
    directory = tmp_path / 'debug'
    session = ImageDebugSession(directory, [image], {})
    notices = []
    progress = ClaudeStreamProgress(lambda *a, **kw: notices.append((a, kw)))
    event = {'type': 'system', 'subtype': 'responses_diagnostic', 'event': 'failed',
             'phase': 'awaiting_headers', 'category': 'HTTP_STATUS', 'http_status': 524,
             'elapsed_s': 120, 'client_request_id': 'test-123'}
    session.consume_claude_line(json.dumps(event), on_event=progress.consume)
    assert json.loads((directory / 'responses_last_request.json').read_text()) == event
    assert json.loads((directory / 'responses_diagnostics.jsonl').read_text()) == event
    assert notices[0][0] == ('responses_api', 'failed', 120)
    assert notices[0][1]['http_status'] == 524
    assert notices[0][1]['api_phase'] == 'awaiting_headers'
    assert 'phase' not in notices[0][1]


def test_retry_discards_finished_tool_item_from_unfinished_response(monkeypatch, capsys):
    partial = sse({'type': 'response.created', 'response': {'id': 'abandoned'}},
                  {'type': 'response.output_item.done', 'output_index': 0,
                   'item': {'id': 'fc_abandoned', 'type': 'function_call',
                            'name': 'rotate_image', 'call_id': 'abandoned_call',
                            'arguments': '{"image_id":"image_0","degrees_clockwise":90}'}})
    bodies = iter([Response(partial), Response(sse({'type': 'response.completed', 'response': FINAL}))])
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(request)
            return next(bodies)

    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: Opener())
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    stats = {'http_attempt_count': 0, 'request_retry_count': 0}
    value = runner.request_with_retries('https://example.invalid/responses', {},
        {'stream': True, 'input': 'inspect'}, runner.time.monotonic() + 30, 2, stats, runner.post)
    assert value == FINAL
    assert 'abandoned' not in json.dumps(value)
    assert len(sent) == 2 and sent[0].data == sent[1].data
    assert sent[0].get_header('X-client-request-id') != sent[1].get_header('X-client-request-id')
    assert stats == {'http_attempt_count': 2, 'request_retry_count': 1}
    events = diagnostics(capsys)
    assert [e['event'] for e in events if e['event'].startswith('retry_')] == ['retry_scheduled', 'retry_recovered']


def test_nested_stream_error_preserves_parameter_failure(monkeypatch):
    install(monkeypatch, Response(sse({'type': 'error', 'error': {
        'type': 'invalid_request_error', 'code': 'unsupported_value',
        'message': 'Unsupported reasoning effort; test-secret'}})))
    with pytest.raises(runner.ResponsesRequestError) as caught:
        post()
    diagnostic = caught.value.diagnostic
    assert diagnostic['upstream_error']['code'] == 'unsupported_value'
    assert 'Unsupported reasoning' in diagnostic['detail']
    assert 'test-secret' not in str(caught.value)
    assert not runner.retryable_request_error(diagnostic)
