"""Host-owned, fixed-destination Codex login relay; never mounts credentials in Agent.

Experimental Codex transport adaptation, not the upstream Claude Messages relay.
ChatGPT login reports token usage but not a dependable USD charge. A USD cap is
therefore rejected, rather than advertised as enforced. Call/time caps are real.
"""
from http.server import BaseHTTPRequestHandler
import http.client
import json
from pathlib import Path
import ssl
import threading
import time
from .session import write_json

MAX_BODY=16*1024*1024
MAX_RESPONSE=64*1024*1024
UPSTREAM_IDLE_SECONDS=300


class Relay:
    def __init__(self, root, auth_file, model, max_calls, deadline, *, usd_budget=None, snapshot=None, provider=None):
        if usd_budget is not None:
            raise ValueError('ChatGPT login does not report enforceable USD costs; use call/time limits, not a fictitious dollar budget')
        if type(max_calls) is not int or max_calls<1:
            raise ValueError('positive call cap required')
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.auth_file=Path(auth_file)
        self.model,self.max_calls,self.deadline=model,max_calls,deadline
        self.lock=threading.Lock()
        self.snapshot=snapshot
        self.before_request=None
        self.provider=provider
        self.billing=provider.billing if provider else 'existing_chatgpt_login'
        path=self.root/'ledger.json'
        self.state=json.loads(path.read_text()) if path.exists() else dict(calls_reserved=0,cost_usd=None,billing=self.billing,requests=[])

    def reserve(self,payload):
        if not isinstance(payload,dict) or payload.get('model')!=self.model:
            raise ValueError('model must equal the operator-selected model')
        if payload.get('stream') is not True or payload.get('store',False) is not False:
            raise ValueError('stream=true and store=false required')
        if not isinstance(payload.get('tools',[]),list):
            raise ValueError('tools must be an array')
        if any(not isinstance(t,dict) or t.get('type') not in ('function','custom','local_shell') for t in payload.get('tools',[])):
            raise ValueError('only client-executed tools permitted; no hosted tools')
        with self.lock:
            if time.monotonic()>=self.deadline or self.state['calls_reserved']>=self.max_calls:
                self.state['last_denial']=('session_time_budget_exhausted' if time.monotonic()>=self.deadline
                                           else 'model_call_budget_exhausted')
                write_json(self.root/'ledger.json',self.state)
                raise PermissionError('model call/time budget exhausted')
            n=self.state['calls_reserved']
            self.state['calls_reserved']+=1
            self.state['requests'].append(dict(id=n,status='reserved',cost_usd=None,started_unix_s=time.time()))
            write_json(self.root/'ledger.json',self.state)
            write_json(self.root/f'{n:04d}-request.json',payload)
        if self.snapshot:
            self.snapshot()
        return n

    def headers(self):
        if self.provider:
            return self.provider.headers()
        # Only the host reads the existing login; do not log values or return them.
        value=json.loads(self.auth_file.read_text())
        tokens=value.get('tokens') or {}
        if value.get('auth_mode')!='chatgpt' or not tokens.get('access_token') or not tokens.get('account_id'):
            raise PermissionError('Existing ChatGPT Codex login required; no credentials are copied into the workspace')
        return {'Authorization':'Bearer '+tokens['access_token'],
                'ChatGPT-Account-Id':tokens['account_id'],'Content-Type':'application/json',
                'Accept':'text/event-stream','originator':'codex_cli_rs',
                'User-Agent':'rax-cloth-codex-adaptation/0.1'}

    def finish(self,n,**details):
        with self.lock:
            self.state['requests'][n].update(details)
            write_json(self.root/'ledger.json',self.state)

    def handler(self):
        relay=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_): pass
            def reply_error(self,status,message):
                body=json.dumps({'error':{'message':message}}).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def do_GET(self): self.reply_error(403,'only POST /responses is allowed')
            def do_POST(self):
                n=None;upstream=None;began=time.monotonic();headers_sent=False;phase='request_validation'
                try:
                    if self.path!='/responses' or self.headers.get('Transfer-Encoding') or self.headers.get('Content-Encoding','identity')!='identity':
                        raise ValueError('route/encoding refused')
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=MAX_BODY: raise ValueError('request size refused')
                    self.connection.settimeout(30)
                    body=self.rfile.read(size)
                    if len(body)!=size: raise ValueError('incomplete request')
                    payload=json.loads(body)
                    if relay.before_request:relay.before_request()
                    auth_headers=relay.headers()
                    n=relay.reserve(payload)
                    timeout=min(UPSTREAM_IDLE_SECONDS,max(.1,relay.deadline-time.monotonic()))
                    relay.finish(n,upstream_idle_timeout_seconds=timeout,phase='connect_and_send')
                    phase='connect_and_send'
                    # Fixed HTTPS destination, no redirects, URL arguments or proxy environment.
                    if relay.provider:
                        upstream=http.client.HTTPSConnection(relay.provider.host,relay.provider.port,
                                                             timeout=timeout,context=ssl.create_default_context())
                        route=relay.provider.path
                    else:
                        upstream=http.client.HTTPSConnection('chatgpt.com',timeout=timeout,context=ssl.create_default_context())
                        route='/backend-api/codex/responses'
                    upstream.request('POST',route,body=body,headers=auth_headers)
                    phase='response_headers'
                    relay.finish(n,phase=phase)
                    response=upstream.getresponse()
                    relay.finish(n,http_status=response.status,headers_elapsed_seconds=time.monotonic()-began)
                    self.send_response(response.status)
                    self.send_header('Content-Type',response.getheader('Content-Type','text/event-stream'))
                    self.send_header('Connection','close');self.end_headers();headers_sent=True
                    self.close_connection=True
                    total=0;usage=None;pending=b''
                    phase='response_stream'
                    relay.finish(n,phase=phase)
                    with (relay.root/f'{n:04d}-response.bin').open('wb') as log:
                        while time.monotonic()<relay.deadline:
                            if upstream.sock:
                                upstream.sock.settimeout(min(UPSTREAM_IDLE_SECONDS,max(.1,relay.deadline-time.monotonic())))
                            chunk=response.read1(65536)
                            if not chunk: break
                            total+=len(chunk)
                            if total>MAX_RESPONSE: raise ValueError('response size limit')
                            log.write(chunk);log.flush()
                            pending+=chunk
                            while b'\n' in pending:
                                line,pending=pending.split(b'\n',1)
                                if line.startswith(b'data: '):
                                    try:
                                        event=json.loads(line[6:]);candidate=event.get('response',{}).get('usage')
                                        if candidate is not None: usage=candidate
                                    except (ValueError,AttributeError): pass
                            self.wfile.write(chunk);self.wfile.flush()
                        else:
                            raise TimeoutError('session wall time exhausted')
                    relay.finish(n,status='returned',http_status=response.status,usage=usage,elapsed_seconds=time.monotonic()-began)
                except Exception as exc:
                    # Do not echo upstream exception contents that could include private headers.
                    if n is not None: relay.finish(n,status='failed',error_type=type(exc).__name__,
                                                  failure_phase=phase,elapsed_seconds=time.monotonic()-began)
                    if not headers_sent:
                        self.reply_error(429 if isinstance(exc,PermissionError) else 502, f'relay refused or failed: {type(exc).__name__}')
                finally:
                    if upstream: upstream.close()
        return Handler
