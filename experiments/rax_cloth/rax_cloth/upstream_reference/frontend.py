"""Host-owned robot/API MCP server: the agent's only path to the robot and to Gemini.
One process per session owns the robot backend (services.backends). It speaks
newline-delimited JSON-RPC (MCP tools) on a private Unix socket, multiplexing the
agent CLI's stdio bridge (harness/claude_cli/mcp_bridge.py) and the agent's own
Python scripts (auto_research_agent/api/runtime.py); every call runs on one
dispatcher thread, so two clients never drive the robot at once. Without --socket it
serves a single client on stdin/stdout.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import sys
import threading
import time
import uuid
from services.backends import make_backend
from services.json_io import jsonable
from .billing import reserve_cost
from .config import Configuration
from . import operator
from .errors import public_failure
from .gemini import GeminiRouter, MAX_REQUEST_BYTES, load_key
from .session import Session
from .timing import Timings

PROTOCOL_VERSION = '2025-06-18'

def tool(name, description, properties=None, required=()):
    return {'name': name, 'description': description, 'inputSchema': {'type': 'object',
            'properties': properties or {}, 'required': list(required), 'additionalProperties': False}}

STATUS = tool('status', 'Session task, wall-clock and Gemini budgets, and the live robot summary.')
OPERATOR = tool('request_operator',
    'Ask the human operator for help or for a scene reset, and wait for the answer. kind "reset": the '
    'operator rearranges the scene (e.g. after a cube fell off the table or to start a new attempt) and '
    'replies when done; "help": a question or a problem you cannot solve through the tools (stuck '
    'object, unclear task, robot fault). Say exactly what you need. The robot does nothing while you wait; '
    'observe again after the answer, because the scene may have changed.',
    {'kind': {'type': 'string', 'enum': ['help', 'reset']},
     'message': {'type': 'string', 'maxLength': 4000},
     'timeout_s': {'type': 'number', 'minimum': 30, 'maximum': 3600,
                   'description': 'How long to wait for the answer (default 900).'}},
    ['kind', 'message'])
SKILL_MERGE = tool('skill_library_merge',
    "Merge your session branch of the skill library (the clone in skill-library/) into the library's main, so "
    'later sessions get your skills. Only committed changes count (git add, git commit first). The harness '
    'checks the format and evidence of every skill you changed (see the continual-learning skill) and merges; '
    'it returns problems to fix, or conflicts to resolve by merging origin/main, or the merged commit.',
    {'message': {'type': 'string', 'maxLength': 500,
                 'description': 'What this merge adds or changes, one line.'}}, ['message'])
GEMINI = {
    'name': 'gemini_generate',
    'description': ('OpenRouter-style chat completion via Gemini 3.8 Flash: text/images, JSON schema output and '
                    'client function calling. Returns the complete assistant message plus usage/cost/API budget; '
                    'Gemini spending is capped for the session. Per-request size/output limits apply. Preserve '
                    'assistant messages including reasoning_details when sending tool results. Tool calls are '
                    'data only: your script validates and executes functions. Images: {type: image, mimeType: '
                    'image/png or image/jpeg, data: base64}, or image_url with an inline data URL. No paths, '
                    'remote URLs or server tools. See the gemini skill references/api.md.'),
    'inputSchema': {'type': 'object', 'additionalProperties': False, 'required': ['messages'], 'properties': {
        'messages': {'type': 'array', 'minItems': 1, 'maxItems': 64, 'items': {'type': 'object'},
                     'description': 'Chat messages: system/user/assistant/tool. Tool results use tool_call_id '
                                    'and string content; preserve complete assistant messages.'},
        'max_tokens': {'type': 'integer', 'minimum': 1, 'maximum': 16384},
        'temperature': {'type': 'number', 'minimum': 0, 'maximum': 2},
        'response_format': {'type': 'object', 'description': 'text, json_object, or json_schema with {name, strict, schema}.'},
        'tools': {'type': 'array', 'maxItems': 64, 'items': {'type': 'object'},
                  'description': 'Client function declarations: {type: function, function: {name, description, parameters, strict?}}.'},
        'tool_choice': {'oneOf': [{'type': 'string', 'enum': ['auto', 'none', 'required']}, {'type': 'object'}],
                        'description': 'Or {type: function, function: {name: declared_name}}.'},
        'reasoning': {'type': 'object', 'description': 'OpenRouter reasoning options: effort or max_tokens, optionally enabled/exclude.'},
    }},
}

class Frontend:
    def __init__(self, session, backend, gemini=None, *, started_at=None):
        self.s, self.backend, self.gemini = session, backend, gemini
        self.started_at = started_at
        self.timings = Timings()
        # Interlocks (arming, workspace limits, holds) belong to the backend itself; errors reach
        # the agent whole (services.controller.errors), never as a label such as "control uncertain".
        self._dispatch = threading.Lock()
        self._backend_tools = {t['name']: t for t in backend.tools()}
        reserved = {'status', 'gemini_generate', 'request_operator', 'skill_library_merge'} & set(self._backend_tools)
        # The persistent skill library clone, when the launcher created one for this workspace.
        self.workspace = Path(session.config.root)/'agent-workspace'
        self.library = (self.workspace/'skill-library'/'.git').is_dir()
        if reserved:
            raise ValueError(f'Backend tools may not reuse harness tool names: {sorted(reserved)}')
    def tools(self):
        return [STATUS, OPERATOR, *self._backend_tools.values(), *([GEMINI] if self.gemini else []),
                *([SKILL_MERGE] if self.library else [])]
    def status(self):
        config = self.s.config
        value = {'task': config.task, 'robot': config.robot, 'observation_profile': config.observation_profile,
                 'api_budget': self.s.budget()}
        if self.started_at is not None:
            value['session_seconds'] = config.session_seconds
            value['seconds_since_launch'] = round(time.time() - self.started_at, 1)
        try:
            value['robot_status'] = self.backend.status()
        except Exception as exc:
            value['robot_status_error'] = public_failure(exc, operation='status')
        return value
    def gemini_call(self, arguments, failure):
        failure['stage'] = 'api_validation'
        payload, reservation = self.gemini.prepare(arguments)
        cost_reservation = reserve_cost(reservation, payload['max_tokens'])
        failure['stage'] = 'api_budget'
        self.s.charge(reservation, cost_reservation)  # Durable reservation before billable I/O.
        failure['stage'] = 'provider_request'
        value, used = self.gemini.execute(payload, reservation)
        failure['stage'] = 'api_accounting'
        self.s.refund(reservation - used, value['usage'], cost_reservation)
        value['budget'] = self.s.budget()
        return value, []
    def call(self, name, arguments):
        """One tools/call; returns an MCP CallToolResult."""
        call_id = uuid.uuid4().hex
        failure = {'stage': 'request'}
        with self._dispatch, self.timings.measure(name if isinstance(name, str) else 'invalid'):
            try:
                if not isinstance(arguments, dict):
                    raise ValueError('Arguments must be an object')
                self.s.audit({'event': 'mcp_request', 'call_id': call_id, 'tool': name, 'arguments': arguments})
                if name == 'status':
                    if arguments:
                        raise ValueError('status takes no arguments')
                    value, images = self.status(), []
                elif name == 'request_operator':
                    failure['stage'] = 'validation'
                    if set(arguments) - {'kind', 'message', 'timeout_s'}:
                        raise ValueError('request_operator takes kind, message, timeout_s')
                    timeout = float(arguments.get('timeout_s', 900))
                    if not 30 <= timeout <= 3600:
                        raise ValueError('timeout_s must be 30..3600')
                    request = operator.open_request(self.s.config.root, arguments.get('kind'), arguments.get('message'))
                    print(f"OPERATOR REQUEST [{request['kind']}] {request['message']}", file=sys.stderr, flush=True)
                    failure['stage'] = 'operator_wait'
                    value = operator.wait(self.s.config.root, request, timeout)
                    try:
                        value['robot_status'] = self.backend.status()
                    except Exception as exc:
                        value['robot_status_error'] = public_failure(exc, operation='status')
                    images = []
                elif name == 'skill_library_merge' and self.library:
                    failure['stage'] = 'validation'
                    if set(arguments) - {'message'} or not str(arguments.get('message', '')).strip():
                        raise ValueError('skill_library_merge takes message (one line)')
                    from services import skill_library
                    value, images = skill_library.merge(self.s.config, self.workspace, arguments['message']), []
                elif name == 'gemini_generate' and self.gemini:
                    value, images = self.gemini_call(arguments, failure)
                elif name in self._backend_tools:
                    failure['stage'] = 'validation'
                    validate = getattr(self.backend, 'validate', None)
                    if validate:
                        validate(name, arguments)
                    failure['stage'] = 'robot_execution'
                    value, images = self.backend.call(name, arguments)
                else:
                    raise PermissionError('Unknown tool')
                value = jsonable(value)
                result = {'content': [{'type': 'text', 'text': json.dumps(value, allow_nan=False)}, *images]}
                self.s.audit({'event': 'mcp_result', 'call_id': call_id, 'tool': name,
                              'result': result['content'][0]['text'][:65536], 'images': len(images)})
                return result
            except Exception as exc:
                # Expected refusals and handled stops carry their own complete message; anything else
                # also gets its traceback tail.
                handled = isinstance(exc, (ValueError, PermissionError)) or getattr(exc, 'handled_stop', False)
                error = jsonable(public_failure(exc, operation=name, stage=failure['stage'],
                                                traceback_tail=not handled))
                try:
                    self.s.audit({'event': 'mcp_error', 'call_id': call_id, **error})
                except Exception:
                    error['audit_recording_failed'] = True
                return {'isError': True, 'content': [{'type': 'text', 'text': json.dumps({'error': error})}]}
    def handle(self, request):
        if not isinstance(request, dict):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
        ident = request.get('id')
        try:
            if request.get('jsonrpc') != '2.0':
                raise ValueError('Invalid JSON-RPC')
            method = request.get('method')
            if method == 'initialize':
                result = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'robot', 'version': '1.0'}}
            elif isinstance(method, str) and method.startswith('notifications/'):
                return None
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': self.tools()}
            elif method == 'tools/call':
                params = request['params']
                result = self.call(params['name'], params.get('arguments', {}))
            else:
                raise PermissionError('Unsupported method')
            return {'jsonrpc': '2.0', 'id': ident, 'result': result}
        except Exception as exc:
            failure = public_failure(exc)
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32000, 'message': failure['reason'], 'data': failure}}
    def _respond(self, line):
        try:
            request = json.loads(line)
        except (ValueError, UnicodeError):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid request'}}
        return self.handle(request)
    def serve_socket(self, listener, stop=None):
        """Multiplex up to 16 clients; requests run one at a time on this thread."""
        listener.setblocking(False)
        buffers = {}
        with selectors.DefaultSelector() as poll:
            poll.register(listener, selectors.EVENT_READ)
            try:
                while not (stop and stop.is_set()):
                    for key, _ in poll.select(timeout=0.5):
                        connection = key.fileobj
                        if connection is listener:
                            try:
                                client, _ = listener.accept()
                            except BlockingIOError:
                                continue
                            if len(buffers) >= 16:
                                client.close()
                                continue
                            client.setblocking(False)
                            buffers[client] = bytearray()
                            poll.register(client, selectors.EVENT_READ)
                            continue
                        try:
                            block = connection.recv(65536)
                            if not block:
                                raise ConnectionError('Client closed')
                            pending = buffers[connection]
                            pending.extend(block)
                            if len(pending) > MAX_REQUEST_BYTES:
                                raise ConnectionError('Request exceeds wire limit')
                            while b'\n' in pending:
                                line, _, tail = pending.partition(b'\n')
                                pending[:] = tail
                                response = self._respond(line)
                                if response is not None:
                                    connection.settimeout(30)
                                    connection.sendall(json.dumps(response, allow_nan=False).encode() + b'\n')
                                    connection.setblocking(False)
                        except (ConnectionError, OSError):
                            poll.unregister(connection)
                            buffers.pop(connection, None)
                            connection.close()
            finally:
                for client in buffers:
                    client.close()
    def serve(self, reader, writer):
        while line := reader.readline(MAX_REQUEST_BYTES + 1):
            if len(line.encode('utf-8')) > MAX_REQUEST_BYTES:
                writer.write(json.dumps({'jsonrpc': '2.0', 'id': None,
                    'error': {'code': -32600, 'message': 'Request exceeds wire limit'}}) + '\n')
                writer.flush()
                return  # Do not interpret the remaining fragment as another request.
            response = self._respond(line)
            if response is not None:
                writer.write(json.dumps(response, allow_nan=False) + '\n')
                writer.flush()
    def close(self):
        try:
            self.backend.close()
        finally:
            self.s.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--socket', type=Path, help='Serve this private Unix socket instead of stdio')
    parser.add_argument('--key-file', type=Path, help='OpenRouter key for gemini_generate (optional)')
    parser.add_argument('--started-at', type=float, help='Launch wall-clock start (epoch seconds)')
    args = parser.parse_args()
    config = Configuration.load(args.config)
    key = load_key(args.key_file) if args.key_file else None
    session = Session(config)
    try:
        backend = make_backend(config)
    except BaseException:
        session.close()
        raise
    frontend = Frontend(session, backend, GeminiRouter(key) if key else None, started_at=args.started_at)
    def stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.socket:
            args.socket.unlink(missing_ok=True)
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(str(args.socket))
                os.chmod(args.socket, 0o600)
                listener.listen(16)
                print(f'MCP listening on {args.socket}', flush=True)
                frontend.serve_socket(listener)
        else:
            frontend.serve(sys.stdin, sys.stdout)
    finally:
        frontend.close()
        if args.socket:
            args.socket.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
