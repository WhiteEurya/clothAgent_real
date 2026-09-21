"""SSH-deployed Responses agent. Only the job's RGB tools are executable.

Each request carries the complete conversation, including reasoning items and
function results. No server-side response storage or CLI token defaults needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Responses endpoint redirected; credential forwarding refused')


def post(url, headers, payload, timeout):
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers=headers, method='POST')
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Error bodies can echo authorization. Never emit them unredacted.
        key = headers.get('Authorization', '').removeprefix('Bearer ')
        detail = redact(exc.read(4000).decode('utf-8', 'replace'), key)
        raise RuntimeError(f'Responses API HTTP {exc.code}: {detail}') from None


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
    seen_responses = set()
    seen_calls = set()
    delivered_image = False
    instructions = (request['system_prompt'] + '\nUse only the supplied RGB function tools. '
                    'Inspect image_0 and relevant supplied images before deciding. Metadata alone '
                    'is not visual evidence. Return one JSON object. '
                    f'At most {limit} tool calls are allowed in this conversation.')
    emit({'type': 'system', 'subtype': 'responses_launch', 'model': request['model'],
          'profile': request['profile'], 'reasoning_effort': request['reasoning_effort'],
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
            response = api_post(url, headers, payload, min(300, remaining))
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
            value = json.loads(text)
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
                  'max_output_tokens': token_limit, 'response_count': responses})
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
            arguments = json.loads(call['arguments'])
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
              'provider': 'responses', 'result': f'{type(exc).__name__}: {redact(exc, os.environ.get("OPENAI_API_KEY"))}'})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
