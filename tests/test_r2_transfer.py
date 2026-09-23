import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloth_agent.planner_backend import RemoteClaudeBackend, PlannerBackendError


def test_r2_is_default_and_missing_config_fails(monkeypatch):
    monkeypatch.setattr('cloth_agent.r2_config.load_r2_env', lambda: None)
    monkeypatch.delenv('R2_ACCOUNT_ID', raising=False)
    backend = RemoteClaudeBackend()
    assert backend.upload_url is None
    with pytest.raises(PlannerBackendError, match='R2_ACCOUNT_ID'):
        backend._upload(Path('image.png'))


def test_r2_put_retry_reuses_key_and_returns_signed_get(monkeypatch):
    for name in ('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET'):
        monkeypatch.setenv(name, 'test-value')
    signed = []
    def sign(operation, **kwargs):
        signed.append((operation, kwargs))
        return 'https://example.invalid/' + operation
    backend = RemoteClaudeBackend()
    backend._r2_client = SimpleNamespace(generate_presigned_url=sign)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] <= 30
        return subprocess.CompletedProcess(command, 28 if len(calls) == 1 else 0, '', 'timeout')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('cloth_agent.planner_backend.time.sleep', lambda _: None)
    assert backend._upload(Path('image.png')) == 'https://example.invalid/get_object'
    assert calls[0] == calls[1]
    assert '--upload-file' in calls[0]
    assert signed[0][1]['Params'] == signed[1][1]['Params']
    assert signed[1][1]['ExpiresIn'] == 3600


def test_env_file_preserves_environment_and_does_not_execute(tmp_path, monkeypatch):
    from cloth_agent.r2_config import load_r2_env
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    path = tmp_path / '.config/clothagent/r2.env'
    path.parent.mkdir(parents=True)
    path.write_text('export R2_BUCKET="from-file"\nR2_ACCOUNT_ID=abc # comment\n')
    monkeypatch.setenv('R2_BUCKET', 'override')
    monkeypatch.delenv('R2_ACCOUNT_ID', raising=False)
    load_r2_env()
    import os
    assert os.environ['R2_BUCKET'] == 'override'
    assert os.environ['R2_ACCOUNT_ID'] == 'abc'


def test_cleanup_only_expired_relay_objects():
    from scripts.cleanup_r2 import cleanup
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    old = 'cloth-agent/' + 'a' * 32 + '.png'
    items = [{'Key': old, 'LastModified': now - timedelta(minutes=61)},
             {'Key': 'cloth-agent/' + 'b' * 32 + '.png', 'LastModified': now - timedelta(minutes=59)},
             {'Key': 'cloth-agent/keep.png', 'LastModified': now - timedelta(days=1)}]
    calls = []
    client = SimpleNamespace(
        get_paginator=lambda _: SimpleNamespace(paginate=lambda **kw: [{'Contents': items}]),
        delete_objects=lambda **kw: calls.append(kw) or {})
    assert cleanup(client, 'bucket', now) == 1
    assert calls[0]['Delete']['Objects'] == [{'Key': old}]
