"""Use the rbs CLI profile in both host and isolated model requests."""
import json
import os
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10 deployment environment
    import tomli as tomllib

PROFILE = 'rbs'


def profile_settings():
    home = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))
    path = home / f'{PROFILE}.config.toml'
    if not path.is_file():
        raise RuntimeError(f'Codex profile missing: {path}')
    base_path = home / 'config.toml'
    base = tomllib.loads(base_path.read_text(encoding='utf-8')) if base_path.is_file() else {}
    selected = tomllib.loads(path.read_text(encoding='utf-8'))
    # Exclude project paths, MCPs, hooks, instructions, trust and tool settings.
    settings = {k: selected.get(k, base.get(k)) for k in
                ('model', 'model_provider', 'model_reasoning_effort', 'service_tier')
                if selected.get(k, base.get(k)) is not None}
    provider = settings.get('model_provider')
    providers = base.get('model_providers', {})
    definition = dict(providers.get(provider, {}))
    definition.update(selected.get('model_providers', {}).get(provider, {}))
    if provider and provider != 'openai' and not definition:
        raise RuntimeError(f'Codex profile {PROFILE} has no definition for provider {provider}')
    if definition:
        allowed = ('name', 'base_url', 'env_key', 'wire_api', 'requires_openai_auth',
                   'request_max_retries', 'stream_max_retries', 'stream_idle_timeout_ms',
                   'supports_websockets')
        if definition.get('http_headers') or definition.get('env_http_headers'):
            raise RuntimeError('Isolated Codex profile custom HTTP headers are not supported')
        settings['model_providers'] = {provider: {k:definition[k] for k in allowed if k in definition}}
    return settings


def _toml_table(values, prefix=()):
    lines = []
    if prefix:
        lines.append('[' + '.'.join(json.dumps(k) for k in prefix) + ']')
    for key, value in values.items():
        if not isinstance(value, dict):
            if not isinstance(value, (str, bool, int, float)):
                raise ValueError('Unsupported Codex provider setting')
            lines.append(json.dumps(key) + ' = ' + json.dumps(value, ensure_ascii=False, allow_nan=False))
    for key, value in values.items():
        if isinstance(value, dict):
            lines += _toml_table(value, (*prefix, key))
    return lines


def isolated_profile_mount(workspace):
    """Mount only model/provider settings, never the user's full config directory."""
    path = Path(workspace) / f'{PROFILE}.config.toml'
    path.write_text('\n'.join(_toml_table(profile_settings())) + '\n', encoding='utf-8')
    return ['--ro-bind', str(path.resolve()), f'/home/agent/.codex/{PROFILE}.config.toml']


def isolated_environment():
    environment = {'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8'}
    settings = profile_settings()
    provider = settings.get('model_providers', {}).get(settings.get('model_provider'), {})
    key = provider.get('env_key')
    if key:
        if key not in os.environ or not os.environ[key]:
            raise RuntimeError(f'Codex profile {PROFILE} requires environment variable {key}')
        environment[key] = os.environ[key]
    return environment
