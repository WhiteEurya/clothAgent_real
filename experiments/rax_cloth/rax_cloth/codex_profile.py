"""Read operator-owned Codex profile transport; never expose auth to the agent."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
try:
    import tomllib
except ImportError:
    import tomli as tomllib


class Profile:
    def __init__(self, home, name):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):
            raise ValueError('invalid Codex profile name')
        home=Path(home)
        base=tomllib.loads((home/'config.toml').read_text())
        path=home/(name+'.config.toml')
        layer=tomllib.loads(path.read_text())
        self.model=layer.get('model',base.get('model'))
        self.effort=layer.get('model_reasoning_effort',base.get('model_reasoning_effort','high'))
        if self.effort not in ('minimal','low','medium','high','xhigh','max','ultra'):
            raise ValueError('unsupported reasoning effort')
        provider=layer.get('model_provider',base.get('model_provider'))
        self.settings={**base.get('model_providers',{}).get(provider,{}),
                       **layer.get('model_providers',{}).get(provider,{})}
        p=self.settings
        if p.get('wire_api','responses')!='responses':
            raise ValueError('profile must use Responses transport')
        if any(p.get(k) for k in ('query_params','http_headers','env_http_headers','requires_openai_auth')):
            raise ValueError('profile transport options unsupported; refusing to silently ignore them')
        url=urlsplit(p.get('base_url',''))
        if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('profile requires a fixed HTTPS base URL without credentials/query')
        self.host=url.hostname;self.port=url.port or 443
        self.path=url.path.rstrip('/')+'/responses'
        self.billing='codex_profile:'+name+'; USD unavailable'
        self.identity={'profile':name,'provider':provider,'base_url':p['base_url'],
                       'model':self.model,'reasoning_effort':self.effort,
                       'profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def headers(self):
        p=self.settings
        if p.get('auth'):
            auth=p['auth']
            try:
                result=subprocess.run([auth['command'],*auth.get('args',[])],
                    cwd=auth.get('cwd'),capture_output=True,text=True,
                    timeout=auth.get('timeout_ms',5000)/1000,check=False)
            except Exception:
                raise PermissionError('Codex profile authentication helper failed') from None
            if result.returncode:
                raise PermissionError('Codex profile authentication helper failed')
            token=result.stdout.strip()
        elif p.get('env_key'):
            token=os.environ.get(p['env_key'],'').strip()
        else:
            token=p.get('experimental_bearer_token','').strip()
        if not token or '\n' in token or '\r' in token:
            raise PermissionError('Codex profile bearer credential unavailable or invalid')
        return {'Authorization':'Bearer '+token,'Content-Type':'application/json','Accept':'text/event-stream'}
