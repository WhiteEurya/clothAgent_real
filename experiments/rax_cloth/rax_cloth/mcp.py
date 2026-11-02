"""Offline stdio protocol adapter. Not a replacement for upstream's full frontend."""
import json
import os
import select
import sys
import time


class Frontend:
    def __init__(self, backend, audit):
        self.backend, self.audit = backend, audit

    def handle(self, request):
        ident = request.get('id')
        began = time.monotonic()
        self.audit('mcp_request', request=request)
        try:
            if request.get('jsonrpc') != '2.0':
                raise ValueError('JSON-RPC 2.0 required')
            method = request.get('method')
            if method == 'notifications/initialized':
                return None
            if method == 'initialize':
                result = dict(protocolVersion='2025-06-18', capabilities={'tools': {}},
                              serverInfo={'name': 'rax-cloth-offline-adapter', 'version': '0.1.0'})
            elif method == 'tools/list':
                result = {'tools': self.backend.tools()}
            elif method == 'ping':
                result = {}
            elif method == 'tools/call':
                p = request['params']
                value, images = self.backend.call(p['name'], p.get('arguments', {}))
                result = {'content': [{'type': 'text', 'text': json.dumps(value, allow_nan=False)}, *images]}
            else:
                raise ValueError('unsupported method')
            response = dict(jsonrpc='2.0', id=ident, result=result)
        except Exception as exc:
            response = dict(jsonrpc='2.0', id=ident, error={'code': -32000, 'message': str(exc)})
        # Keep complete content including saved image references and returned image bytes.
        self.audit('mcp_response', response=response, seconds=time.monotonic()-began)
        return response

    def serve(self, stop=None, deadline=None):
        pending = b''
        try:
            while not (stop and stop.is_set()):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    break
                pending += chunk
                if len(pending) > 1_000_000:
                    raise ValueError('request too large')
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    try:
                        response = self.handle(json.loads(line))
                    except Exception as exc:
                        response = dict(jsonrpc='2.0', id=None, error={'code': -32700, 'message': str(exc)})
                    if response is not None:
                        print(json.dumps(response, allow_nan=False), flush=True)
        finally:
            self.backend.close()
