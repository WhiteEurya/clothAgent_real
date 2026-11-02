"""Codex CLI adaptation for official RAX; offline by default, explicitly supervised real mode.

Original exploration/compile/memory workflow, new CLI + Responses transport.
This module never imports a robot SDK or executes a generated program on the host.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import sys
import subprocess
import tempfile
import threading
import time
from .backend import ClothBackend
from .official import configure, prepare
from .responses_relay import Relay
from .progress import Progress
from .session import write_json
from .upstream import ROOT

PROMPT='''Read AGENTS.md, TASK.md and the universal RAX skills for your current step.
This is a short OFFLINE integration episode. Observe the supplied RGB through MCP.
Work autonomously using the original RAX workflow: investigate the available evidence,
write a Python controller under code/<project>/controller.py defining main(ctx),
read fresh observations and compute any proposals from them, test its perception and
control-interface behavior inside your workspace, then improve code if tests reveal
an issue. Use the official api.runtime.Context interface for robot tools.
All motion tools refuse physical execution. Replay is historical evidence, not an
answer to a new action. Do not claim a manipulation outcome, improvement or success.
Record your code tests, limitations and next needed evidence in memory/PROGRESS.md
and memory/RUNS.md. Keep learned skills draft and evidence-labeled; no verified skill
can come from mock/replay. Do not import any existing garment algorithm or prompt.
Finish after this small integration episode; report exact code and test commands.
'''

REAL_PROMPT='''Read AGENTS.md, TASK.md and the official universal RAX skills. This is a
supervised real session for T-shirt flattening. Start with tools/list, status and fresh
stationary RGB observations. Develop your own strategy from current evidence; no garment
policy is supplied. Follow the original explore, compile, evaluate and improve workflow.
All physical actions must use the protected MCP tools and require a fresh observation_id.
Only the onsite operator can authorize actions. A refusal is not permission to bypass
the service. Never import SDKs, tune safety constraints or automatically restart STOP.
Save code, evidence and memory throughout the session. Use fresh observations in generated
programs. Evaluate actual outcomes honestly and keep unverified skills draft.
End with a clear report of what was executed, observed, learned and remains unverified.
'''


def cli_config(model, effort='high'):
    if not re.fullmatch(r'[a-zA-Z0-9_.:/-]{1,120}',model):
        raise ValueError('invalid model id')
    if effort not in ('minimal','low','medium','high','xhigh','max','ultra'):
        raise ValueError('invalid reasoning effort')
    return f'''model = {json.dumps(model)}
model_provider = "rax_offline"
model_reasoning_effort = {json.dumps(effort)}
approval_policy = "never"
web_search = "disabled"
sandbox_mode = "workspace-write"
[sandbox_workspace_write]
# Permit Unix MCP connections inside the mandatory outer --unshare-all namespace.
# This grants no host/robot network access; never use this config outside that namespace.
network_access = true
[features]
multi_agent = false
[model_providers.rax_offline]
name = "RAX-Cloth host model relay"
base_url = "http://127.0.0.1:18765"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 360000
[mcp_servers.robot]
command = "/usr/bin/python3"
args = ["/adapter/mcp_bridge.py", "/channels/mcp.sock"]
startup_timeout_sec = 30
tool_timeout_sec = 3600
default_tools_approval_mode = "approve"
'''


def isolated_command(checkout, workspace, channels, frames, private, model, seed=0):
    """Adapt the existing project empty-root bwrap layout; additionally unshare network.

    No root bind, production checkout, inherited home, auth file or controller SDK.
    A Unix model socket is the ONLY egress; the host relay has one fixed endpoint.
    """
    binary=shutil.which('codex');bwrap=shutil.which('bwrap')
    if not binary or not bwrap or not Path('/usr/bin/socat').is_file():
        raise RuntimeError('codex, bubblewrap and /usr/bin/socat required')
    package=Path(binary).resolve().parent.parent
    if not (package/'bin/codex-code-mode-host').is_file():
        raise RuntimeError('full standalone Codex package required')
    command=[bwrap,'--die-with-parent','--new-session','--unshare-all','--tmpfs','/',
             '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/opt',
             '--ro-bind',str(package),'/opt/codex-package',
             '--bind',str(workspace),'/workspace',
             '--ro-bind',str(channels),'/channels','--ro-bind',str(frames),'/frames',
             '--dir','/adapter','--dir','/home','--dir','/home/agent',
             '--bind',str(private/'codex-home'),'/home/agent/.codex',
             '--ro-bind',str(private/'codex-config.toml'),'/home/agent/.codex/config.toml',
             '--ro-bind',str(checkout/'harness/claude_cli/mcp_bridge.py'),'/adapter/mcp_bridge.py']
    for file in ('inner.py','probe.py'):
        command+=['--ro-bind',str(ROOT/'agent_assets'/file),'/adapter/'+file]
    for file in ('/usr','/bin','/lib','/lib64','/etc/alternatives','/etc/ld.so.cache'):
        if Path(file).exists(): command+=['--ro-bind',str(Path(file).resolve()),file]
    # Both callers supply a fresh env to Popen/run. Older host bwrap has no --clearenv.
    command+=['--setenv','HOME','/home/agent','--setenv','CODEX_HOME','/home/agent/.codex',
              '--setenv','PATH','/opt/codex-package/bin:/opt/codex-package/codex-path:/usr/bin:/bin',
              '--setenv','LANG','C.UTF-8','--setenv','RAX_MCP_SOCKET','/channels/mcp.sock',
              '--setenv','RAX_AGENT_MODEL',model,'--setenv','PYTHONPATH','/workspace:/workspace/.pylib',
              '--setenv','PYTHONHASHSEED',str(seed),'--chdir','/workspace']
    return command


def probe(command):
    native_net=os.readlink('/proc/self/ns/net')
    result=subprocess.run(command+['/usr/bin/python3','-I','/adapter/probe.py',native_net],
                          capture_output=True,text=True,timeout=20,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
    return dict(passed=result.returncode==0,returncode=result.returncode,
                stdout=result.stdout[-4000:],stderr=result.stderr[-4000:])


class AgentOfflineBackend(ClothBackend):
    def __init__(self,config,frames):
        super().__init__(config)
        self.frames=Path(frames)
        self.sequence=max((int(p.name) for p in (self.root/'observations').glob('[0-9]*')),default=-1)+1
        self.frame_calls=0
    def call(self,name,arguments):
        if name=='observe':
            if self.frame_calls>=128: raise PermissionError('observation call limit reached')
            self.frame_calls+=1
        value,images=super().call(name,arguments)
        if name=='observe':
            target=self.frames/f'{self.sequence:06d}.png'
            shutil.copyfile(value['image_path'],target)
            value={**value,'image_path':'/frames/'+target.name}
        return value,images


def archive_programs(workspace,root):
    """Read code as data using no-follow directory handles; never import/execute it."""
    rows=[];limit=128
    def walk(fd,prefix='',depth=0):
        if depth>8:return
        for name in sorted(os.listdir(fd)):
            if len(rows)>=limit:return
            try:
                child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            except OSError:continue
            try:
                info=os.fstat(child)
                relative=prefix+name
                if stat.S_ISDIR(info.st_mode):walk(child,relative+'/',depth+1)
                elif stat.S_ISREG(info.st_mode) and name.endswith('.py') and info.st_size<=1_000_000:
                    data=b''
                    while len(data)<=1_000_000:
                        block=os.read(child,65536)
                        if not block:break
                        data+=block
                    if len(data)>1_000_000:continue
                    digest=hashlib.sha256(data).hexdigest()
                    path=Path(root)/'programs'/digest
                    path.mkdir(parents=True,exist_ok=True)
                    (path/'source.py').write_bytes(data)
                    rows.append(dict(file='code/'+relative,sha256=digest,physical_verified=False,
                                     test_evidence='See immutable model/agent transcripts; not host-executed'))
            finally:os.close(child)
    try:fd=os.open(Path(workspace)/'code',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    except OSError:return []
    try:walk(fd)
    finally:os.close(fd)
    with (Path(root)/'program-history.jsonl').open('a') as out:
        out.write(json.dumps(dict(unix_s=time.time(),programs=rows))+'\n')
    return rows


def parse_events(path):
    from cloth_agent.codex_event_status import completion_status
    events=[];thread_id=None;answer=False
    for line in Path(path).read_text(errors='replace').splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if not isinstance(event,dict):continue
        events.append(event)
        if event.get('type')=='thread.started':thread_id=event.get('thread_id')
        item=event.get('item') or {}
        if isinstance(item,dict) and item.get('type')=='agent_message' and str(item.get('text','')).strip():
            answer=True
    return dict(thread_id=thread_id,has_answer=answer,**completion_status(events))


def recover_interrupted(state, private, max_seconds):
    """A killed launcher cannot refund time or lose the durable CLI thread id."""
    if state.get('status') != 'running' or not state['attempts']:
        return
    row = state['attempts'][-1]
    state['elapsed_seconds'] += min(max_seconds, max(0, time.time()-row['started_unix_s']))
    events = Path(private)/row['directory']/'events.jsonl'
    if events.is_file():
        state['thread_id'] = parse_events(events)['thread_id'] or state.get('thread_id')
    row['status'] = state['status'] = 'interrupted'


def completion_outcome(succeeded, parsed, ledger, max_calls, timed_out=False):
    if succeeded:
        return 'agent_completed', None
    errors=json.dumps(parsed.get('errors',[]))
    if ledger.get('last_denial')=='model_call_budget_exhausted':
        return 'model_call_budget_exhausted', 'Configured model-call cap reached; this is not provider rate limiting.'
    if timed_out or ledger.get('last_denial')=='session_time_budget_exhausted':
        return 'session_time_budget_exhausted', 'Configured session time cap reached.'
    return 'interrupted_or_failed', errors[:2000]


def relay_budget_status(relay):
    return {'calls_used':relay.state['calls_reserved'],
            'calls_limit':relay.max_calls,
            'calls_remaining':max(0,relay.max_calls-relay.state['calls_reserved']),
            'reported_cost_usd':None,'session_cost_limit_usd':None,
            'billing':relay.billing,
            'seconds_remaining':max(0,relay.deadline-time.monotonic())}


def launch(args):
    onsite=getattr(args,'onsite',False)
    initial_reset=getattr(args,'initial_reset',False)
    if initial_reset and not onsite:raise ValueError('--initial-reset requires --onsite')
    if onsite:
        if args.mode!='real_supervised' or getattr(args,'workflow','upstream')!='upstream':
            raise ValueError('--onsite requires real_supervised and upstream workflow')
        if not args.preflight_only and not sys.stdin.isatty():
            raise PermissionError('--onsite requires a foreground operator terminal')
    if args.budget_usd is not None:
        raise ValueError('Dollar cap unavailable for this transport; use --max-model-calls and --max-seconds. Costs will be null, not zero.')
    profile_name=getattr(args,'profile','chatgpt')
    provider=None
    if profile_name!='chatgpt':
        from .codex_profile import Profile
        provider=Profile(Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex'))),profile_name)
    if args.model is None:
        from cloth_agent.codex_astra_pipeline import MODEL
        args.model=provider.model if provider else MODEL
    workflow=getattr(args,'workflow','offline_smoke')
    if workflow=='upstream':
        from . import baseline
        checkout,config=baseline.configure(args.session,args.mode,args.replay_manifest,getattr(args,'robot_config',None))
        prepare_workspace=baseline.prepare
    else:
        if args.mode=='real_supervised':raise ValueError('custom real-service deployment retired; use --workflow upstream --robot-config')
        checkout,config=configure(args.session,args.mode,args.replay_manifest)
        prepare_workspace=prepare
    config=replace(config,agent_model=args.model,max_model_calls=args.max_model_calls,
                   session_seconds=args.max_seconds)
    if onsite:
        config=replace(config,backend_options={**config.backend_options,'manual_reset':True},
            task_description=config.task_description+'\nOnsite console: the operator started this supervised session. '
            'Motion permission is supplied by the host for this session; do not request a separate arm step. '
            'Environment reset is manual: use request_operator(kind="reset") and wait for the operator reply. '
            'Do not call xarm_reset in this session. Stop/fault/safety checks still apply.')
        if initial_reset:
            config=replace(config,task_description=config.task_description+
                '\nBefore this Agent starts, the host executes the existing mechanical pickup/drop reset. '
                'The starting scene is already reset; read a fresh image and begin the task from that scene. '
                'Do not request another initial reset merely because no operator confirmation appears in this conversation. '
                'Subsequent evaluation resets can still be requested from the onsite operator.')
    from services.controller.session import Session
    from services.controller.frontend import Frontend
    from harness.claude_cli.relay_common import UnixHTTPServer
    owner=Session(config)
    stop=threading.Event();process=None;listener=None;server=None;frontend=None
    workers=[];started=time.monotonic();previous={};relay=None;attempt=None;backend=None;console=None
    private=config.root/'codex-adaptation';private.mkdir(exist_ok=True)
    progress=Progress(private,getattr(args,'progress_interval',10))
    statefile=private/'agent-state.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else {'attempts':[], 'thread_id':None, 'elapsed_seconds':0}
    identity=dict(mode=args.mode,model=args.model,seed=args.seed,max_model_calls=args.max_model_calls,
                  max_seconds=args.max_seconds,upstream_commit=json.loads((ROOT/'upstream.json').read_text())['commit'],
                  replay_manifest=config.backend_options.get('replay_manifest_sha256'))
    if provider:
        identity['codex_profile']=provider.identity
    if workflow=='upstream':
        identity['workflow']='upstream'
        if onsite:identity['onsite_console']=True
        if initial_reset:identity['initial_reset']=True
        if args.mode=='real_supervised':
            identity['robot_config']=config.backend_options['robot_config']
            identity['robot_config_sha256']=hashlib.sha256(Path(identity['robot_config']).read_bytes()).hexdigest()
    identityfile=private/'identity.json'
    try:
        progress.emit('准备官方 RAX 工作空间；模式='+args.mode+'，profile='+profile_name+'，模型='+args.model)
        if identityfile.exists() and json.loads(identityfile.read_text())!=identity:
            raise ValueError('resume identity/budgets differ; use the same configuration')
        write_json(identityfile,identity)
        if state['attempts'] and not args.resume:
            raise ValueError('session already attempted model work; use --resume')
        recover_interrupted(state,private,args.max_seconds)
        write_json(statefile,state)
        workspace=prepare_workspace(config)
        # Generated workspace only; official source remains unchanged.
        instructions=workspace/'AGENTS.md'
        marker='# Codex CLI adaptation\n'
        overlay=private/'workspace-overlay.json'
        if not overlay.exists():
            if state['attempts']:
                raise ValueError('missing operator-owned workspace provenance; cannot rewrite agent files on resume')
            instructions.write_text(marker+'\nThis session uses Codex CLI, not upstream Claude Code. '
                'The original RAX workflow/skills follow below. Python: /usr/bin/python3; '
                'NumPy and OpenCV are available. Observations are under read-only /frames. '
                'Only supplied local data is available; general network access is absent. '
                'Use api.runtime.Context through /channels/mcp.sock. No robot SDK exists here.\n\n'+instructions.read_text())
            write_json(overlay,{'file':'AGENTS.md','sha256':hashlib.sha256(instructions.read_bytes()).hexdigest(),
                                'change':'Codex transport, installed tools and offline execution context preamble'})
        home=private/'codex-home';home.mkdir(exist_ok=True)
        (private/'codex-config.toml').write_text(cli_config(args.model,provider.effort if provider else 'high'))
        (workspace/'api').mkdir(exist_ok=True)
        # Protect the official Python API from agent edits via a separate read-only bind below.
        frames=config.root/'agent-frames';frames.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='rax-channels-') as temporary:
            channels=Path(temporary)
            command=isolated_command(checkout,workspace,channels,frames,private,args.model,args.seed)
            if workflow=='upstream':
                # Expose exactly the shared library under a sandbox-local path;
                # no host parent directories or robot project become visible.
                from services.skill_library import library_path
                library=library_path(config).resolve()
                command+=['--ro-bind',str(library),'/skill-library-origin',
                          '--setenv','RAX_SHARED_LIBRARY','/skill-library-origin']
            if provider:
                command+=['--ro-bind',str(private/'codex-config.toml'),
                          '/home/agent/.codex/'+profile_name+'.config.toml',
                          '--setenv','RAX_CODEX_PROFILE',profile_name]
            command+=['--ro-bind',str(checkout/'auto_research_agent/api'),'/workspace/api']
            progress.emit('检查文件、机器人 SDK 和网络隔离')
            report=probe(command)
            write_json(private/'isolation.json',report)
            write_json(private/'launch-command.json',command)
            if not report['passed']:
                progress.emit('隔离检查失败；详见 codex-adaptation/isolation.json')
                state['status']='ISOLATION_FAILED';write_json(statefile,state)
                raise PermissionError('Agent isolation preflight failed; no model or hardware started. See '+str(private/'isolation.json'))
            if args.preflight_only:
                return {'status':'PREFLIGHT_PASSED','model_started':False,'hardware_connected':False}
            remaining=args.max_seconds-state['elapsed_seconds']
            if remaining<=0: raise PermissionError('session time budget exhausted')
            deadline=time.monotonic()+remaining
            auth=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
            relay=Relay(private/'model',auth,args.model,args.max_model_calls,deadline,
                        snapshot=lambda:archive_programs(workspace,private),provider=provider)
            progress.emit('隔离通过；检查宿主模型认证')
            relay.headers() # Validate host login presence only; no remote request yet.
            if workflow=='upstream':
                from services.backends import make_backend
                backend=make_backend(config)
            else:backend=AgentOfflineBackend(config,frames)
            class OfflineFrontend(Frontend):
                def status(self):
                    value=super().status()
                    value['api_budget']=relay_budget_status(relay)
                    return value
                def call(self,name,arguments):
                    if (name=='request_operator' and console is not None and isinstance(arguments,dict)
                            and arguments.get('kind') in ('help','reset')
                            and isinstance(arguments.get('message'),str) and arguments['message'].strip()
                            and len(arguments['message'])<=4000):
                        console.pause()
                        # Wait for the onsite reply within the existing session
                        # budget, rather than an agent-selected 30-second reset.
                        arguments={**arguments,'timeout_s':min(3600,max(30,deadline-time.monotonic()))}
                    if name=='request_operator' and args.mode!='real_supervised':
                        self.s.audit({'event':'offline_operator_request','arguments':arguments})
                        return {'content':[{'type':'text','text':json.dumps({'status':'OPERATOR_UNAVAILABLE','scene_reset':False})}]}
                    return super().call(name,arguments)
            frontend=OfflineFrontend(owner,backend,gemini=None,started_at=time.time())
            if onsite:
                from .onsite_console import OnsiteConsole
                console=OnsiteConsole(config.root,backend,stop,deadline,progress)
                relay.before_request=console.wait_for_operator
                console.enable()
                progress.emit('现场模式已启动：本轮可执行，reset 由你完成并按回车继续。Ctrl+C 或 stop 停止。','operator_notice')
            elif args.mode=='real_supervised':
                progress.emit('实机 session 已建立，当前未授权运动。request_operator 的请求会显示在此终端；限时 arm 仍由现场操作员设置。','operator_notice')
            frontend.library=workflow=='upstream' or args.mode=='real_supervised'
            listener=socket.socket(socket.AF_UNIX);listener.bind(str(channels/'mcp.sock'));listener.listen(8)
            os.chmod(channels/'mcp.sock',0o600)
            server=UnixHTTPServer(str(channels/'model.sock'),relay.handler())
            os.chmod(channels/'model.sock',0o600)
            for target in (lambda:frontend.serve_socket(listener,stop),server.serve_forever):
                worker=threading.Thread(target=target,daemon=True);worker.start();workers.append(worker)
            if args.resume and state['thread_id'] is None:
                raise ValueError('no recorded Codex thread id to resume')
            attempt=private/f'episode_{len(state["attempts"]):04d}'
            attempt.mkdir(exist_ok=False)
            row={'directory':attempt.name,'status':'running','started_unix_s':time.time()}
            state['attempts'].append(row);state['status']='running';write_json(statefile,state)
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
                previous[sig]=signal.signal(sig,lambda *_:stop.set())
            if initial_reset:
                from .initial_reset import run as run_initial_reset
                run_initial_reset(config.root,backend,console,stop,deadline,args.seed,progress)
            command+=['/usr/bin/python3','/adapter/inner.py']
            if args.resume:command+=[state['thread_id']]
            if workflow=='upstream':
                from harness.claude_cli.launcher import RESUME_PROMPT
                prompt=RESUME_PROMPT if args.resume else (workspace/'START_PROMPT.md').read_text().strip()
            else:prompt=PROMPT
            (attempt/'prompt.txt').write_text(prompt)
            with (attempt/'events.jsonl').open('w') as events,(attempt/'stderr.txt').open('w') as errors:
                progress.start(attempt/'events.jsonl',relay,deadline,args.max_seconds)
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=events,stderr=errors,
                                         text=True,start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
                process.stdin.write(prompt);process.stdin.close()
                while process.poll() is None and not stop.wait(.2):
                    if console is not None:console.poll()
                    else:progress.operator_requests(config.root)
                    progress.tick()
                    if time.monotonic()>=deadline:
                        stop.set();break
                if args.mode=='real_supervised':
                    # Stop service before process cleanup, code archiving or result generation.
                    backend.close()
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                progress.tick(force=True)
            parsed=parse_events(attempt/'events.jsonl')
            old_thread=state['thread_id']
            state['thread_id']=parsed['thread_id'] or state['thread_id']
            succeeded=(process.returncode==0 and parsed['final_turn_completed'] and not parsed['errors']
                       and (not args.resume or parsed['thread_id']==old_thread)
                       and parsed['has_answer'])
            status,reason=completion_outcome(succeeded,parsed,relay.state,args.max_model_calls,
                                             time.monotonic()>=deadline)
            row.update(status=status,reason=reason,returncode=process.returncode,completion=parsed)
            state['status']=row['status']
            programs=archive_programs(workspace,private)
            try:robot_status=backend.status()
            except Exception as exc:
                robot_status={'physical_actions':None,'safety':'UNKNOWN','status_error':type(exc).__name__}
            result=dict(status=state['status'],reason=reason,events_path=str(attempt/'events.jsonl'),
                        harness='RAX-Codex/xArm' if workflow=='upstream' else 'legacy offline integration diagnostic',
                        workflow=workflow,
                        model=args.model,mode=args.mode,physical_actions=robot_status['physical_actions'],
                        robot_status=robot_status,task_success=None,programs=programs,
                        llm_calls_reserved=relay.state['calls_reserved'],cost_usd=None,
                        billing=relay.billing,
                        elapsed_seconds=time.monotonic()-started,thread_id=state['thread_id'])
            write_json(attempt/'result.json',result)
            progress.emit('运行结束：'+status,'finished')
            (config.root/'summary.md').write_text('# RAX-Cloth Codex attempt\n\n'+json.dumps(result,indent=2)+'\n\nTask success is not automatically certified. See model requests/responses, MCP audit and operator-service evidence.\n')
            return result
    finally:
        progress.close()
        stop.set()
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        if server:server.shutdown();server.server_close()
        for worker in workers:worker.join(timeout=2)
        if listener:listener.close()
        if frontend:frontend.close()
        else:
            try:
                if backend:backend.close()
            finally:owner.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        if attempt is not None:
            state['elapsed_seconds']+=time.monotonic()-started
            write_json(statefile,state)


def main():
    from cloth_agent.codex_astra_pipeline import MODEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--mode',choices=['mock','replay','real_supervised'],default='replay')
    parser.add_argument('--workflow',choices=['upstream','offline_smoke'],default='upstream',
                        help='Original RAX START_PROMPT/skills by default; offline_smoke is the old short diagnostic')
    parser.add_argument('--robot-config',type=Path,help='Existing robot config required for real_supervised')
    parser.add_argument('--replay-manifest',type=Path,default=ROOT.parent/'configs/replay.example.json')
    parser.add_argument('--model',default=None)
    parser.add_argument('--profile',default='chatgpt',help='Default chatgpt uses local Codex ChatGPT login; api explicitly selects the local API profile')
    parser.add_argument('--max-model-calls',type=int,default=8)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--budget-usd',type=float)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--onsite',action='store_true',help='Foreground supervised session: startup enables motion; manual resets acknowledged in this terminal')
    parser.add_argument('--initial-reset',action='store_true',help='Execute existing protected mechanical reset once before starting Agent; requires --onsite')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--progress-interval',type=int,default=10,help='Progress heartbeat interval in seconds (default 10)')
    args=parser.parse_args()
    if not 1<=args.progress_interval<=300:
        parser.error('--progress-interval must be 1..300 seconds')
    if args.max_model_calls<1 or not 1<=args.max_seconds<=28800 or not 0<=args.seed<=4294967295:
        parser.error('positive call cap, 1..28800 seconds and uint32 seed required')
    from .control_service import ServiceUnavailable
    try:result=launch(args)
    except ServiceUnavailable as exc:
        parser.exit(2,str(exc)+'\n')
    print(json.dumps(result,indent=2))
    if result['status'] not in ('agent_completed','PREFLIGHT_PASSED'):
        raise SystemExit(2)


if __name__=='__main__':main()
