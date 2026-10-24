"""Existing Codex multimodal command interface, enclosed in a data-blind namespace.

No host root, project files, run folders, tools, session history or MCP config is
mounted. Each request gets only its stage-filtered images, prompt and schema.
Bubblewrap failure is fatal: never fall back to a leaky host workspace.
"""
from __future__ import annotations
import os
import json
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from PIL import Image
from cloth_agent.codex_astra_pipeline import build_command, MODEL
from .discovery_data import read, save
from .model_errors import ModelInvocationError


def isolated_command(binary, workspace, auth=None):
    bwrap=shutil.which('bwrap')
    if not bwrap:raise RuntimeError('bubblewrap required for stage isolation; no unsafe fallback')
    # Static Codex release in this project: no system shell or Python is exposed.
    command=[bwrap,'--die-with-parent','--new-session','--unshare-user','--unshare-pid',
             '--unshare-ipc','--unshare-uts','--tmpfs','/',
             '--dir','/opt','--ro-bind',str(Path(binary).resolve()),'/opt/codex',
             '--bind',str(workspace),'/work','--dir','/home','--dir','/home/agent',
             '--dir','/home/agent/.codex','--tmpfs','/tmp','--proc','/proc','--dev','/dev',
             '--dir','/etc','--setenv','HOME','/home/agent',
             '--setenv','CODEX_HOME','/home/agent/.codex','--setenv','PATH','/opt',
             '--setenv','SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt','--chdir','/work']
    for name in ('/etc/ssl','/etc/resolv.conf','/etc/hosts'):
        if Path(name).exists():command+=['--ro-bind',str(Path(name).resolve()),name]
    if auth and Path(auth).is_file():command+=['--ro-bind',str(Path(auth).resolve()),'/home/agent/.codex/auth.json']
    # Never inherit proxy variables, project/environment instructions or other secrets.
    return command


class CodexMultimodal:
    def __init__(self, model=MODEL, timeout=900, binary='codex', timing_output=None):
        self.model=model;self.timeout=timeout;self.binary=shutil.which(binary)
        self.timing_output=Path(timing_output) if timing_output is not None else None
        self.invocations=0
        if not self.binary:raise RuntimeError('Codex CLI missing')

    def invoke(self, prompt, images, schema):
        invocation_started=time.monotonic()
        self.invocations+=1
        with tempfile.TemporaryDirectory(prefix='decision_state_') as tmp:
            work=Path(tmp);save(work/'schema.json',schema)
            local_images=[];image_dimensions=[]
            for i,image in enumerate(images):
                p=work/f'image_{i:03d}.png'
                # RGB pixels only: remove paths, EXIF, embedded text/metadata.
                with Image.open(image) as im:
                    image_dimensions.append(list(im.size))
                    im.convert('RGB').save(p)
                local_images.append(Path('/work')/p.name)
            cmd=build_command('/opt/codex',Path('/work'),Path('/work'),read_only=True)
            cmd[cmd.index('--model')+1]=self.model
            cmd[-1:-1]=['--skip-git-repo-check','--ephemeral','--ignore-user-config','--ignore-rules',
                        '-c','features.shell_tool=false','-c','web_search="disabled"']
            for image in local_images:cmd[-1:-1]=['--image',str(image)]
            auth=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
            command=isolated_command(self.binary,work,auth)+cmd
            started=time.monotonic()
            with (work/'events.jsonl').open('w') as events,(work/'stderr.txt').open('w') as errors:
                timeline=None
                if self.timing_output is not None:
                    from .model_timing import EventTimeline
                    timeline=EventTimeline(work/'events.jsonl',started)
                    timeline.start()
                try:
                    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=events,stderr=errors,
                                       text=True,start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
                except BaseException:
                    if timeline is not None:timeline.finish()
                    raise
                try:p.communicate(prompt,timeout=self.timeout)
                except BaseException as exc:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    if isinstance(exc,subprocess.TimeoutExpired):
                        raise ModelInvocationError('Isolated model request timed out',retryable=True,details={'timeout_s':self.timeout}) from exc
                    raise
                finally:
                    process_wall_s=time.monotonic()-started
                    if timeline is not None:
                        from .model_timing import summarize
                        observed=timeline.finish()
                        save(self.timing_output/f'invocation_{self.invocations:03d}.json',dict(
                            host_setup_s=started-invocation_started, process_wall_s=process_wall_s,
                            image_count=len(images), image_dimensions=image_dimensions,
                            model=self.model, returncode=p.returncode, observer_error=timeline.error,
                            event_timing=observed, summary=summarize(observed,process_wall_s)))
            events_parsed=[]
            usage={};tool_calls=0;visual_tool_calls=0;completed_turns=0
            for line in (work/'events.jsonl').read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                events_parsed.append(event)
                item=event.get('item',{})
                if event.get('type')=='item.completed' and item.get('type') in (
                        'command_execution','mcp_tool_call','web_search','image_view','tool_call'):
                    tool_calls+=1
                    if item.get('type')=='image_view' or 'view_image' in str(item.get('tool','')):
                        visual_tool_calls+=1
                if event.get('type')=='turn.completed':
                    completed_turns+=1
                    for k,v in event.get('usage',{}).items():
                        if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
            if p.returncode:
                stderr=(work/'stderr.txt').read_text()[-1600:]
                failures=[e for e in events_parsed if e.get('type') in ('error','turn.failed')]
                details=dict(returncode=p.returncode,stderr=stderr,failure_events=failures[-4:])
                # Retry only the isolated model invocation, never a robot dispatch.
                # Namespace/setup errors stay fatal and never switch to an unisolated transport.
                retryable='bwrap:' not in stderr.lower()
                raise ModelInvocationError('isolated Codex failed: '+json.dumps(details,ensure_ascii=False),retryable=retryable,details=details)
            from cloth_agent.codex_event_status import completion_status
            completion=completion_status(events_parsed)
            if completion['errors'] or not completion['final_turn_completed']:
                raise RuntimeError('Codex turn failed or incomplete: '+json.dumps(completion,ensure_ascii=False))
            answer=work/'answer.txt'
            if not answer.is_file() or not answer.read_text().strip():raise RuntimeError('No model answer')
            return dict(text=answer.read_text(),usage=usage,elapsed_s=time.monotonic()-started,
                        recovered_errors=completion['recovered_errors'],
                        tool_calls=tool_calls,visual_tool_calls=visual_tool_calls,completed_turns=completed_turns,
                        image_count=len(local_images),isolation='bubblewrap-allowlist',model=self.model)
