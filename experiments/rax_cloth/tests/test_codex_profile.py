import json
import time
from pathlib import Path
from unittest.mock import Mock
import pytest
from rax_cloth.codex_profile import Profile
from rax_cloth.codex_agent import cli_config
from rax_cloth import responses_relay as transport
from test_codex_agent import invoke,payload


def profile(tmp_path):
    (tmp_path/'config.toml').write_text('model="base"\nmodel_reasoning_effort="medium"\n')
    token=tmp_path/'token';token.write_text('fake-api-token')
    (tmp_path/'api.config.toml').write_text('model="test-model"\nmodel_provider="test"\nmodel_reasoning_effort="high"\n'
        '[model_providers.test]\nbase_url="https://provider.example/v1"\nwire_api="responses"\n'
        '[model_providers.test.auth]\ncommand="/usr/bin/cat"\nargs=['+json.dumps(str(token))+']\ntimeout_ms=5000\n')
    return Profile(tmp_path,'api')


def test_local_profile_auth_stays_on_host(tmp_path):
    p=profile(tmp_path)
    assert p.model=='test-model' and p.effort=='high'
    assert p.headers()['Authorization']=='Bearer fake-api-token'
    assert 'fake-api-token' not in json.dumps(p.identity)+cli_config(p.model,p.effort)
    assert 'provider.example' not in cli_config(p.model,p.effort)


def test_profile_relay_uses_selected_endpoint_not_chatgpt(tmp_path,monkeypatch):
    p=profile(tmp_path)
    r=transport.Relay(tmp_path/'relay',tmp_path/'absent-auth.json',p.model,2,time.monotonic()+60,provider=p)
    response=Mock(status=200);response.getheader.return_value='text/event-stream'
    response.read1.side_effect=[b'data: {"response":{"usage":{"input_tokens":2}}}\n\n',b'']
    conn=Mock();conn.getresponse.return_value=response
    factory=Mock(return_value=conn);monkeypatch.setattr(transport.http.client,'HTTPSConnection',factory)
    assert b'200 OK' in invoke(r,payload())
    assert factory.call_args.args==('provider.example',443)
    assert conn.request.call_args.args==('POST','/v1/responses')
    assert conn.request.call_args.kwargs['headers']['Authorization']=='Bearer fake-api-token'
    assert r.state['billing'].startswith('codex_profile:api')
    assert all(b'fake-api-token' not in f.read_bytes() for f in r.root.iterdir())


def test_missing_profile_fails_without_login_fallback(tmp_path):
    (tmp_path/'config.toml').write_text('model="base"')
    with pytest.raises(FileNotFoundError):Profile(tmp_path,'api')


def test_profile_invalid_auth_not_exposed(tmp_path):
    p=profile(tmp_path);(tmp_path/'token').unlink()
    with pytest.raises(PermissionError,match='authentication helper failed'):p.headers()


def test_header_timeout_is_logged_and_bounded_by_session_deadline(tmp_path,monkeypatch):
    p=profile(tmp_path)
    r=transport.Relay(tmp_path/'relay',tmp_path/'absent',p.model,2,time.monotonic()+120,provider=p)
    conn=Mock();conn.getresponse.side_effect=TimeoutError()
    factory=Mock(return_value=conn);monkeypatch.setattr(transport.http.client,'HTTPSConnection',factory)
    assert b'502' in invoke(r,payload())
    assert 45<factory.call_args.kwargs['timeout']<=120
    assert r.state['requests'][0]['failure_phase']=='response_headers'
    assert r.state['calls_reserved']==1
    assert factory.call_count==1  # no hidden retry or duplicate provider charge


def test_long_reasoning_wait_uses_300_second_cap(tmp_path,monkeypatch):
    p=profile(tmp_path)
    r=transport.Relay(tmp_path/'relay',tmp_path/'absent',p.model,2,time.monotonic()+1200,provider=p)
    conn=Mock();conn.getresponse.side_effect=TimeoutError()
    factory=Mock(return_value=conn);monkeypatch.setattr(transport.http.client,'HTTPSConnection',factory)
    invoke(r,payload())
    assert factory.call_args.kwargs['timeout']==300
