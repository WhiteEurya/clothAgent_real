"""SSH-deployed Responses agent. Only the job's RGB tools are executable.

Each request carries the complete conversation, including reasoning items and
function results. No server-side response storage or CLI token defaults needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import random
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def emit(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def redact(text, key):
    text = str(text).replace(key, '[REDACTED]') if key else str(text)
    return re.sub(r'sk-[A-Za-z0-9_*.-]+', '[REDACTED]', text)


def load_provider(profile):
    """Read only provider/credential routing; never log or copy credentials."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise RuntimeError('Remote Python 3.10 requires tomli>=2; install it in CLOTH_REMOTE_IMAGE_PYTHON') from None
    root = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
    config = tomllib.loads((root / 'config.toml').read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', profile):
        raise ValueError('invalid provider profile name')
    overlay_path = root / f'{profile}.config.toml'
    overlay = (tomllib.loads(overlay_path.read_text()) if overlay_path.is_file()
               else config.get('profiles', {}).get(profile))
    if not isinstance(overlay, dict):
        raise ValueError('provider profile not found')
    name = overlay.get('model_provider', config.get('model_provider'))
    provider = dict(config.get('model_providers', {}).get(name, {}))
    provider.update(overlay.get('model_providers', {}).get(name, {}))
    if provider.get('wire_api') != 'responses':
        raise ValueError('profile must select a Responses provider')
    base = provider.get('base_url', '').rstrip('/')
    url = urllib.parse.urlsplit(base)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('provider base_url must be an HTTPS origin/path without credentials')
    env_key = provider.get('env_key')
    key = os.environ.get(env_key, '') if isinstance(env_key, str) else ''
    if not key:
        raise ValueError('provider credential environment variable is missing')
    headers = dict(provider.get('http_headers', {}))
    for header, variable in provider.get('env_http_headers', {}).items():
        value = os.environ.get(variable)
        if not value:
            raise ValueError('provider header environment variable is missing')
        headers[header] = value
    headers.update({'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    return base + '/responses', headers, key


class EndpointRedirectError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EndpointRedirectError('Responses endpoint redirected; credential forwarding refused')


class ResponsesRequestError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"Responses API {diagnostic['category']} at {diagnostic['phase']} "
            f"after {diagnostic['elapsed_s']:.3f}s "
            f"(client_request_id={diagnostic['client_request_id']}): {diagnostic.get('detail', '')}")


class ResponsesResult(dict):
    """Keep local transport metadata out of the API response/conversation."""
    def __init__(self, value, diagnostic):
        super().__init__(value)
        self.diagnostic = dict(diagnostic)


def validation_error(response, category, detail):
    diagnostic = dict(getattr(response, 'diagnostic', {}))
    diagnostic.update(event='failed', category=category, phase='validating_response', detail=detail,
                      response_id=response.get('id'), client_request_id=diagnostic.get('client_request_id'),
                      elapsed_s=diagnostic.get('elapsed_s', 0))
    emit({'type': 'system', 'subtype': 'responses_diagnostic', **diagnostic})
    raise ResponsesRequestError(diagnostic)


class RequestTrace:
    """Only transport metadata and redacted errors; never prompts/images/deltas."""
    def __init__(self, headers, payload, byte_count, timeout):
        self.started = time.monotonic()
        self.deadline = self.started + timeout
        self.last_notice = self.started
        self.secrets = [value.removeprefix('Bearer ') for name, value in headers.items()
                        if name.lower() not in {'content-type', 'accept', 'x-client-request-id'}
                        and isinstance(value, str) and value]
        self.data = dict(client_request_id=headers['X-Client-Request-Id'],
                         stream_requested=payload.get('stream', False), request_bytes=byte_count,
                         request_timeout_s=timeout, http_status=None, response_headers={},
                         headers_received_s=None, first_body_read_s=None, first_event_s=None,
                         last_event=None, event_count=0, received_bytes=0,
                         last_body_read_s=None, phase='awaiting_headers')

    def clean(self, value):
        if isinstance(value, dict):
            return {k: self.clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, '[REDACTED]')
            return redact(value, '')
        return value

    def notice(self, event, **fields):
        self.data.update(self.clean(fields))
        self.data.update(elapsed_s=round(time.monotonic() - self.started, 3), event=event,
                         timestamp=datetime.now(timezone.utc).isoformat())
        emit({'type': 'system', 'subtype': 'responses_diagnostic', **self.data})
        self.last_notice = time.monotonic()

    def fail(self, category, detail, **fields):
        self.notice('failed', category=category, detail=self.clean(str(detail))[:4000], **fields)
        raise ResponsesRequestError(dict(self.data))

    def headers(self, response):
        allowed = ('x-request-id', 'request-id', 'cf-ray', 'server', 'content-type',
                   'retry-after', 'openai-processing-ms', 'x-envoy-upstream-service-time')
        self.notice('headers_received', http_status=response.code,
                    headers_received_s=round(time.monotonic() - self.started, 3),
                    response_headers={k: response.headers[k][:512] for k in allowed
                                      if response.headers.get(k)})

    def read(self, response, *, line=False, size=65536):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.fail('REQUEST_DEADLINE', 'Client request time budget exhausted')
        # urllib's socket timeout alone is an inactivity timeout. Bound each
        # read by the remaining request budget too (CPython HTTPResponse).
        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        if sock is not None:
            sock.settimeout(remaining)
        block = response.readline(size) if line else response.read1(size)
        if time.monotonic() > self.deadline:
            self.fail('REQUEST_DEADLINE', 'Client request time budget exhausted')
        if block:
            elapsed = round(time.monotonic() - self.started, 3)
            self.data['received_bytes'] += len(block)
            self.data['last_body_read_s'] = elapsed
            if self.data['first_body_read_s'] is None:
                self.notice('first_body_read', first_body_read_s=elapsed)
        return block


def response_error(value):
    error = value.get('error') or {}
    return {k: error[k] for k in ('type', 'code', 'message') if k in error} if isinstance(error, dict) else {}


def output_summary(items):
    if not isinstance(items, list):
        return []
    return [{'type': item.get('type'), 'id': item.get('id'),
             'call_id': item.get('call_id'), 'name': item.get('name'),
             'argument_chars': len(item.get('arguments') or ''),
             'text_chars': sum(len(block.get('text') or '') for block in (item.get('content') or [])
                               if isinstance(block, dict))}
            for item in items if isinstance(item, dict)]


def check_response(value, trace):
    if not isinstance(value, dict):
        trace.fail('INVALID_RESPONSE', 'Expected a Responses object')
    status = value.get('status')
    details = value.get('incomplete_details') or {}
    reason = details.get('reason') if isinstance(details, dict) else None
    fields = dict(response_id=value.get('id'), response_status=status,
                  incomplete_reason=reason, upstream_error=response_error(value),
                  output_summary=output_summary(value.get('output') or []))
    if status != 'completed':
        category = 'OUTPUT_TOKEN_LIMIT' if reason == 'max_output_tokens' else 'RESPONSE_NOT_COMPLETED'
        trace.fail(category, reason or status or 'Missing response status', **fields)
    if not isinstance(value.get('output'), list) or not isinstance(value.get('id'), str) or not value['id']:
        trace.fail('INVALID_RESPONSE', 'Expected output list and nonempty response id', **fields)
    trace.notice('completed', **fields)
    return ResponsesResult(value, trace.data)


def read_sse(response, trace):
    data = []
    event_bytes = 0
    done_items = {}
    started_items = {}
    response_id = None
    # A terminal event contains the complete response, including reasoning.
    # Never execute argument deltas or accept partial output at EOF.
    while True:
        line = trace.read(response, line=True, size=8 * 1024 * 1024 + 1)
        if not line:
            trace.fail('STREAM_EOF', 'Stream ended before a terminal Responses event')
        event_bytes += len(line)
        if event_bytes > 8 * 1024 * 1024:
            trace.fail('INVALID_STREAM', 'SSE event exceeds 8 MiB')
        text = line.decode('utf-8').rstrip('\r\n')
        if text.startswith('data:'):
            data.append(text[5:].removeprefix(' '))
        elif not text:
            event_bytes = 0
            if not data:
                continue
            raw = '\n'.join(data)
            data = []
            if raw == '[DONE]':
                trace.fail('STREAM_EOF', '[DONE] arrived without a terminal Responses event')
            event = json.loads(raw)
            if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                trace.fail('INVALID_STREAM', 'SSE data lacks an event type')
            kind = event['type']
            trace.data.update(last_event=kind, event_count=trace.data['event_count'] + 1)
            if trace.data['first_event_s'] is None:
                trace.notice('first_event', first_event_s=round(time.monotonic() - trace.started, 3))
            if kind == 'response.created':
                response_id = (event.get('response') or {}).get('id')
                trace.data['response_id'] = response_id
            if kind == 'response.output_item.added':
                index = event.get('output_index')
                item = event.get('item')
                if type(index) is not int or index < 0 or not isinstance(item, dict) or index in started_items:
                    trace.fail('INVALID_STREAM', 'Invalid/duplicate output_item.added')
                started_items[index] = item.get('id')
            if kind == 'response.output_item.done':
                index = event.get('output_index')
                item = event.get('item')
                if type(index) is not int or index < 0 or not isinstance(item, dict) or index in done_items:
                    trace.fail('INVALID_STREAM', 'Invalid/duplicate output_item.done')
                if index in started_items and started_items[index] != item.get('id'):
                    trace.fail('INVALID_STREAM', 'Output item identity changed during stream')
                done_items[index] = item
                trace.notice('output_item_done', done_output_summary=output_summary(
                    [done_items[k] for k in sorted(done_items)]))
            if kind in {'response.completed', 'response.failed', 'response.incomplete'}:
                value = event.get('response')
                if not isinstance(value, dict) or value.get('status') != kind.split('.')[1]:
                    trace.fail('INVALID_STREAM', 'Terminal event/status mismatch')
                if response_id is not None and response_id != value.get('id'):
                    trace.fail('INVALID_STREAM', 'Response identity changed during stream')
                # RBS was observed to send complete output_item.done objects but
                # an empty output array at response.completed. Recover only
                # finished items, after the successful terminal event. Deltas
                # and unfinished items are never executable.
                if kind == 'response.completed' and value.get('output') == [] and done_items:
                    if (set(done_items) != set(range(len(done_items))) or
                            not set(started_items) <= set(done_items) or
                            len({item.get('id') for item in done_items.values()}) != len(done_items) or
                            any(not item.get('id') or item.get('status', 'completed') != 'completed'
                                for item in done_items.values())):
                        trace.fail('INVALID_STREAM', 'Terminal output missing and streamed items incomplete')
                    value = {**value, 'output': [done_items[k] for k in sorted(done_items)]}
                    trace.notice('terminal_output_reconstructed', terminal_output_empty=True,
                                 output_source='output_item.done')
                else:
                    trace.data['output_source'] = 'terminal_response'
                return check_response(value, trace)
            if kind == 'error':
                # RBS also returns {type:error,error:{type,code,message}}.
                # Keep nested provider details; never classify an unknown
                # stream error as transient based on HTTP 200 alone.
                error = response_error(event) or {k: event[k] for k in ('type', 'code', 'message') if k in event}
                trace.fail('STREAM_API_ERROR', error.get('message', 'API stream error'), upstream_error=error)
        # Comments/keepalives are network activity, not evidence of model progress.
        if time.monotonic() - trace.last_notice >= 10:
            trace.notice('body_progress')


def post(url, headers, payload, timeout):
    headers = {**headers, 'X-Client-Request-Id': headers.get('X-Client-Request-Id') or str(uuid.uuid4()),
               'Accept': 'text/event-stream' if payload.get('stream') else 'application/json'}
    body = json.dumps(payload, ensure_ascii=False).encode()
    trace = RequestTrace(headers, payload, len(body), timeout)
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    trace.notice('started')
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            trace.headers(response)
            content_type = response.headers.get('content-type', '').split(';')[0].lower().strip()
            if content_type == 'text/event-stream':
                trace.notice('body_started', phase='reading_sse', response_format='sse')
                return read_sse(response, trace)
            # Some gateways ignore stream=true. Record this explicitly and
            # still demand a completed, valid JSON response.
            trace.notice('body_started', phase='reading_json', response_format='json',
                         stream_fallback=bool(payload.get('stream')))
            chunks = []
            while True:
                block = trace.read(response)
                if not block:
                    break
                chunks.append(block)
                if trace.data['received_bytes'] > 8 * 1024 * 1024:
                    trace.fail('INVALID_RESPONSE', 'JSON response exceeds 8 MiB')
            return check_response(json.loads(b''.join(chunks)), trace)
    except ResponsesRequestError:
        raise
    except EndpointRedirectError as exc:
        trace.fail('HTTP_REDIRECT', str(exc))
    except urllib.error.HTTPError as exc:
        trace.headers(exc)
        trace.notice('http_error', phase='reading_http_error')
        try:
            detail = trace.read(exc, size=4000).decode('utf-8', 'replace')
            try:
                value = json.loads(detail)
                error = response_error(value) if isinstance(value, dict) else {}
            except ValueError:
                error = {}
            trace.data['upstream_error'] = trace.clean(error)
        except ResponsesRequestError:
            # Preserve the known HTTP status even if reading its body times out.
            detail = 'HTTP error body exceeded request deadline'
        except (OSError, ValueError, http.client.HTTPException) as body_error:
            detail = 'HTTP error body unreadable: ' + type(body_error).__name__
        finally:
            exc.close()
        trace.fail('HTTP_STATUS', f'HTTP {exc.code}: {detail}',
                   interpretation='Endpoint returned an HTTP error; internal upstream cause is not established')
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        category = ('DNS_ERROR' if isinstance(reason, socket.gaierror) else
                    'TLS_ERROR' if isinstance(reason, ssl.SSLError) else
                    'CLIENT_TIMEOUT' if isinstance(reason, TimeoutError) else 'NETWORK_ERROR')
        trace.fail(category, str(reason), exception_type=type(reason).__name__)
    except (ValueError, UnicodeError) as exc:
        # JSON decoder errors may include input text: keep the class only.
        trace.fail('INVALID_STREAM' if trace.data['phase'] == 'reading_sse' else 'INVALID_RESPONSE',
                   type(exc).__name__)


MAX_REQUEST_RETRIES = 2
RETRY_HTTP_STATUSES = {500, 502, 503, 504, 520, 522, 524}


def retryable_request_error(diagnostic):
    # Explicit protocol/model/credential failures must not be hidden by retries.
    error = diagnostic.get('upstream_error') or {}
    permanent = {'invalid_request_error', 'authentication_error', 'invalid_api_key',
                 'permission_error', 'insufficient_quota', 'max_output_tokens',
                 'unsupported_value', 'invalid_value'}
    if error.get('type') in permanent or error.get('code') in permanent:
        return False
    category = diagnostic.get('category')
    return (category in {'STREAM_EOF', 'CLIENT_TIMEOUT'} or
            category == 'HTTP_STATUS' and diagnostic.get('http_status') in RETRY_HTTP_STATUSES or
            category in {'STREAM_API_ERROR', 'RESPONSE_NOT_COMPLETED'} and
            (error.get('code') or error.get('type')) in {
                'service_unavailable', 'service_unavailable_error', 'server_is_overloaded', 'overloaded_error'} or
            category == 'NETWORK_ERROR' and diagnostic.get('exception_type') in {
                'ConnectionResetError', 'ConnectionAbortedError', 'BrokenPipeError',
                'RemoteDisconnected', 'IncompleteRead'})


def retry_delay(diagnostic, retry_number):
    delay = 2 ** retry_number + random.uniform(0, .5)
    header = (diagnostic.get('response_headers') or {}).get('retry-after')
    if header is not None:
        try:
            seconds = float(header)
        except (ValueError, TypeError):
            try:
                date = parsedate_to_datetime(header)
                seconds = date.timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                return delay
        if math.isfinite(seconds):
            delay = max(delay, seconds)
        else:
            return math.inf
    return delay


def request_with_retries(url, headers, payload, deadline, request_index, stats, api_post):
    """Replay one read-only model request, never a tool or an entire job.

    The snapshot excludes unfinished output; tools/history are owned by run().
    The same deadline covers all attempts and backoff. Provider billing/state
    is not exactly-once: each replay has a fresh client request ID.
    """
    snapshot = json.dumps(payload, ensure_ascii=False)
    fingerprint = hashlib.sha256(snapshot.encode()).hexdigest()
    previous_id = None
    for attempt in range(1, MAX_REQUEST_RETRIES + 2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Responses conversation deadline exceeded before API attempt')
        identity = str(uuid.uuid4())
        fields = dict(request_index=request_index, attempt=attempt,
                      max_attempts=MAX_REQUEST_RETRIES + 1, client_request_id=identity,
                      previous_client_request_id=previous_id, payload_sha256=fingerprint)
        emit({'type': 'system', 'subtype': 'responses_diagnostic',
              'event': 'attempt_started', 'phase': 'request_attempt', **fields})
        stats['http_attempt_count'] += 1
        if attempt > 1:
            stats['request_retry_count'] += 1
        try:
            response = api_post(url, {**headers, 'X-Client-Request-Id': identity},
                                json.loads(snapshot), min(300, remaining))
        except ResponsesRequestError as exc:
            diagnostic = {**exc.diagnostic, **fields}
            can_retry = retryable_request_error(diagnostic)
            delay = retry_delay(diagnostic, attempt) if can_retry and attempt <= MAX_REQUEST_RETRIES else None
            remaining = deadline - time.monotonic()
            decision = ('not_retryable' if not can_retry else
                        'attempts_exhausted' if attempt > MAX_REQUEST_RETRIES else
                        'retry_after_too_long' if delay > 60 else
                        'conversation_deadline' if remaining <= delay + 1 else 'retry_scheduled')
            diagnostic.update(event=decision, retry_decision=decision,
                              retry_delay_s=delay if delay is not None and math.isfinite(delay) else None,
                              conversation_remaining_s=round(max(0, remaining), 3))
            emit({'type': 'system', 'subtype': 'responses_diagnostic', **diagnostic})
            if decision != 'retry_scheduled':
                raise ResponsesRequestError(diagnostic) from None
            previous_id = identity
            time.sleep(delay)
            continue
        if attempt > 1:
            emit({'type': 'system', 'subtype': 'responses_diagnostic',
                  **getattr(response, 'diagnostic', {}), **fields,
                  'event': 'retry_recovered', 'phase': 'request_attempt'})
        return response


def text_of(response):
    return '\n'.join(block['text'] for item in response['output']
                     if item.get('type') == 'message'
                     for block in item.get('content', []) if block.get('type') == 'output_text').strip()


def model_content(result):
    content = []
    for block in result['content']:
        if block['type'] == 'text':
            content.append({'type': 'input_text', 'text': block['text']})
        elif block['type'] == 'image':
            content.append({'type': 'input_image',
                            'image_url': 'data:' + block['mimeType'] + ';base64,' + block['data']})
        else:
            raise ValueError('unsupported tool result content')
    return content


def run(job, request, prompt, api_post=post):
    from image_tools import ImageTools, IMAGE_TOOLS, TOOLS, read_hook, orientation_guard
    from codex_adapter import CodexEvents, output_schema

    url, headers, key = load_provider(request['profile'])
    deadline = time.monotonic() + request.get('timeout_s', 900)
    limit = request['max_tool_calls']
    token_limit = request.get('max_output_tokens', 32768)
    if type(token_limit) is not int or token_limit < 16:
        raise ValueError('max_output_tokens must be an integer >=16')
    tools = ImageTools(job, request['image_count'], edit_limit=request.get('image_edit_limit'),
                       call_limit=min(64, limit), result_byte_limit=900000)
    definitions = {tool['name']: tool for tool in TOOLS}
    state = CodexEvents(limit)
    history = [{'role': 'user', 'content': prompt}]
    pending_delivery = []
    usage = {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0}
    responses = 0
    request_stats = {'http_attempt_count': 0, 'request_retry_count': 0}
    seen_responses = set()
    seen_calls = set()
    delivered_image = False
    instructions = (request['system_prompt'] + '\nUse only the supplied RGB function tools. '
                    'Inspect image_0 and relevant supplied images before deciding. Metadata alone '
                    'is not visual evidence. Return one JSON object. '
                    f'At most {limit} tool calls are allowed in this conversation.')
    emit({'type': 'system', 'subtype': 'responses_launch', 'model': request['model'],
          'profile': request['profile'], 'reasoning_effort': request['reasoning_effort'],
          'api_stream': request.get('api_stream', True),
          'max_request_retries': MAX_REQUEST_RETRIES,
          'max_output_tokens': token_limit, 'endpoint': url,
          'conversation': 'full_history', 'delivery_boundary': 'API accepted request bytes'})

    def translated(event):
        emit({'type': 'system', 'subtype': 'responses_event', 'responses_event': event})
        for message in state.consume(event):
            emit(message)

    while responses <= limit:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Responses conversation deadline exceeded')
        payload = {'model': request['model'], 'instructions': instructions,
                   'stream': request.get('api_stream', True),
                   'reasoning': {'effort': request['reasoning_effort']},
                   'max_output_tokens': token_limit, 'store': False,
                   'include': ['reasoning.encrypted_content'], 'input': history,
                   'tools': [{'type': 'function', 'name': t['name'], 'description': t['description'],
                              'parameters': t['inputSchema']} for t in TOOLS],
                   'parallel_tool_calls': False,
                   'text': {'format': {'type': 'json_schema', 'name': 'planner_result',
                                       'strict': True, 'schema': output_schema(request['schema'])}}}
        emit({'type': 'system', 'subtype': 'responses_request', 'request_index': responses + 1,
              'max_output_tokens': token_limit, 'conversation_items': len(history)})
        try:
            response = request_with_retries(url, headers, payload, deadline,
                                            responses + 1, request_stats, api_post)
        except ResponsesRequestError:
            raise
        except Exception as exc:
            raise RuntimeError(redact(exc, key)) from None
        responses += 1
        if not isinstance(response, dict) or not isinstance(response.get('output'), list):
            raise ValueError('invalid Responses response')
        if response.get('status') != 'completed':
            reason = (response.get('incomplete_details') or {}).get('reason')
            raise RuntimeError('Responses response did not complete: ' + str(reason or response.get('status')))
        identity = response.get('id')
        if not identity or identity in seen_responses:
            raise ValueError('missing/duplicate Responses response id')
        seen_responses.add(identity)
        measured = response.get('usage') or {}
        if measured.get('output_tokens', 0) > token_limit:
            emit({'type': 'system', 'subtype': 'responses_budget_warning',
                  'requested_max_output_tokens': token_limit,
                  'reported_output_tokens': measured['output_tokens'],
                  'reason': 'Provider reported output exceeding the requested budget'})
        for field in ('input_tokens', 'output_tokens'):
            usage[field] += measured.get(field, 0)
        usage['reasoning_tokens'] += (measured.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        emit({'type': 'system', 'subtype': 'responses_completed', 'response_id': identity,
              'request_index': responses, 'usage': measured,
              'max_output_tokens': response.get('max_output_tokens')})
        # Only after a completed API response do we report the exact submitted
        # tool bytes as delivered. This does not prove model understanding.
        for item in pending_delivery:
            read_hook(job, {'hook_event_name': 'PostToolUse',
                           'tool_name': 'mcp__cloth_image__' + item['tool'],
                           'tool_use_id': item['id'], 'tool_input': item['arguments'],
                           'tool_response': item['result']})
            translated({'type': 'item.completed', 'item': item})
            if not item['result'].get('isError') and any(
                    block.get('type') == 'image' for block in item['result']['content']):
                delivered_image = True
        pending_delivery = []
        output = response['output']
        if any(item.get('type') not in {'message', 'reasoning', 'function_call'} for item in output):
            raise ValueError('unexpected Responses output/tool type')
        calls = [item for item in output if item.get('type') == 'function_call']
        if not calls:
            text = text_of(response)
            try:
                value = json.loads(text)
            except ValueError:
                validation_error(response, 'FINAL_JSON_INVALID',
                                 f'Completed response contains no valid final JSON (text_chars={len(text)})')
            if not isinstance(value, dict):
                raise ValueError('Responses final is not a JSON object')
            if not delivered_image:
                raise ValueError('Responses final has no RGB tool evidence')
            translated({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text}})
            translated({'type': 'turn.completed', 'usage': usage})
            final = state.final(0)
            if request.get('orientation_correction'):
                orientation_guard(job, {'hook_event_name': 'Stop', 'last_assistant_message': text})
            emit({**final, 'provider': 'responses', 'model': request['model'],
                  'profile': request['profile'], 'reasoning_effort': request['reasoning_effort'],
                  'max_output_tokens': token_limit, 'response_count': responses, **request_stats,
                  'usage_scope': 'completed_responses_only; failed attempts may incur unreported usage'})
            return
        history.extend(output)
        if len(calls) + len(seen_calls) > limit:
            raise ValueError('Responses tool-call budget exceeded')
        for call in calls:
            call_id, name = call.get('call_id'), call.get('name')
            if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
                raise ValueError('missing/duplicate function call id')
            if name not in definitions:
                raise ValueError('unexpected Responses function tool')
            seen_calls.add(call_id)
            try:
                arguments = json.loads(call['arguments'])
            except (KeyError, TypeError, ValueError):
                validation_error(response, 'TOOL_ARGUMENTS_INVALID',
                                 'Completed function call has missing/invalid JSON arguments')
            if not isinstance(arguments, dict):
                raise ValueError('function arguments must be an object')
            item = {'id': call_id, 'type': 'mcp_tool_call', 'server': 'cloth_image',
                    'tool': name, 'arguments': arguments}
            translated({'type': 'item.started', 'item': item})
            try:
                value = tools.call(name, arguments)
                result = (tools.image_result(value) if name in IMAGE_TOOLS else
                          {'content': [{'type': 'text', 'text': json.dumps(value)}]})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                result = {'isError': True, 'content': [{'type': 'text', 'text': redact(exc, key)}]}
            item.update(status='completed', result=result)
            pending_delivery.append(item)
            history.append({'type': 'function_call_output', 'call_id': call_id,
                            'output': model_content(result)})
    raise ValueError('Responses request budget exceeded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args(argv)
    job = args.job.resolve(strict=True)
    try:
        request = json.loads((job / 'codex_request.json').read_text())
        run(job, request, sys.stdin.read())
        return 0
    except Exception as exc:
        emit({'type': 'result', 'subtype': 'error_responses', 'is_error': True,
              'provider': 'responses', 'diagnostic': getattr(exc, 'diagnostic', None),
              'result': f'{type(exc).__name__}: {redact(exc, os.environ.get("OPENAI_API_KEY"))}'})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
