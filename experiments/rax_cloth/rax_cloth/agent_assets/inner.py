"""Runs inside the empty-root network namespace. No host credentials or robot SDK."""
import os
import signal
import socket
import subprocess
import sys
import time

proxy=subprocess.Popen(['/usr/bin/socat','TCP-LISTEN:18765,bind=127.0.0.1,reuseaddr,fork',
                        'UNIX-CONNECT:/channels/model.sock'],stdout=sys.stderr,stderr=sys.stderr)
agent=None
try:
    for _ in range(50):
        if proxy.poll() is not None: raise RuntimeError('isolated model forwarding process failed')
        try:
            s=socket.create_connection(('127.0.0.1',18765),timeout=.1);s.close();break
        except OSError: time.sleep(.05)
    else: raise RuntimeError('isolated forwarding process did not become ready')
    command=['/opt/codex-package/bin/codex','exec','--model',os.environ['RAX_AGENT_MODEL'],
             '--cd','/workspace','--json','--color','never','--sandbox','workspace-write',
             '--skip-git-repo-check','--ignore-rules','--output-last-message','/workspace/output/answer.txt']
    if os.environ.get('RAX_CODEX_PROFILE'):
        command+=['--profile',os.environ['RAX_CODEX_PROFILE']]
    if len(sys.argv)>1:
        command+=['resume',sys.argv[1]]
    command+=['-']
    agent=subprocess.Popen(command,stdin=sys.stdin,stdout=sys.stdout,stderr=sys.stderr)
    sys.exit(agent.wait())
finally:
    if agent and agent.poll() is None: agent.terminate()
    proxy.terminate()
    try: proxy.wait(timeout=2)
    except subprocess.TimeoutExpired: proxy.kill()
