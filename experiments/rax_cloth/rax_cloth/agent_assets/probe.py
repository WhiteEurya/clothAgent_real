"""Isolation acceptance checks inside exactly the Agent mount/network policy."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

assert os.readlink('/proc/self/ns/net')!=sys.argv[1], 'host network namespace exposed'
for module in ('cloth_agent','xarm','pyrealsense2'):
    assert importlib.util.find_spec(module) is None, 'robot module exposed: '+module
for path in ('/mnt/newssd','/home/sja','/root/.ssh','/home/agent/.codex/auth.json','/dev/video0'):
    assert not Path(path).exists(), 'host data/device exposed: '+path
import numpy
import cv2
p=Path('/workspace/.isolation-probe');p.write_text('ok');p.unlink()
try:
    Path('/usr/.rax-write-test').write_text('bad')
except OSError:
    pass
else:
    raise RuntimeError('system mount writable')
try:
    Path('/home/agent/.codex/config.toml').write_text('bad')
except OSError:
    pass
else:
    raise RuntimeError('operator CLI config writable')
library=os.environ.get('RAX_SHARED_LIBRARY')
if library:
    # Rewrite the deployment path inside isolation, never run git with the
    # agent-owned clone/config on the host. Merge still uses official validation.
    subprocess.run(['git','-C','/workspace/skill-library','config','remote.origin.url',library],
                   check=True,capture_output=True,timeout=10)
    subprocess.run(['git','-C','/workspace/skill-library','fetch','--no-tags','origin'],
                   check=True,capture_output=True,timeout=10)
    try:
        Path(library+'/.rax-write-test').write_text('probe')
    except OSError:
        pass
    else:
        Path(library+'/.rax-write-test').unlink()
        raise RuntimeError('shared library must be read-only to agent')
print(json.dumps({'isolated':True,'numpy':numpy.__version__,'opencv':cv2.__version__,
                  'robot_sdk_importable':False,'host_network_visible':False,
                  'skill_library_fetch_verified':bool(library)}))
