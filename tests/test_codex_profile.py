import json
from pathlib import Path

import pytest

from cloth_agent import codex_profile as profile
from cloth_agent.codex_astra_pipeline import build_command


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home=tmp_path/'codex';home.mkdir()
    (home/'config.toml').write_text('''model_provider="other"
[model_providers.rbs_test]
name="fixture"
base_url="https://example.invalid/v1"
env_key="RBS_FIXTURE_KEY"
wire_api="responses"
[mcp_servers.secret]
command="do-not-copy"
''')
    (home/'rbs.config.toml').write_text('''model="gpt-6-astra"
model_provider="rbs_test"
model_reasoning_effort="high"
[projects.private]
trust_level="trusted"
''')
    monkeypatch.setenv('CODEX_HOME',str(home))
    monkeypatch.setenv('RBS_FIXTURE_KEY','fixture-only')
    monkeypatch.setenv('UNRELATED_SECRET','must-not-inherit')
    return home


def test_host_command_selects_rbs():
    cmd=build_command('codex',Path('/work'),Path('/out'),read_only=True)
    assert cmd[:4]==['codex','-p','rbs','exec']
    assert '--sandbox' in cmd and cmd[-1]=='-'


def test_namespace_profile_and_environment_are_allowlisted(profile_home,tmp_path):
    work=tmp_path/'work';work.mkdir()
    mount=profile.isolated_profile_mount(work)
    saved=profile.tomllib.loads((work/'rbs.config.toml').read_text())
    assert saved['model_provider']=='rbs_test'
    assert saved['model_providers']['rbs_test']['wire_api']=='responses'
    assert 'mcp_servers' not in saved and 'projects' not in saved
    assert 'fixture-only' not in (work/'rbs.config.toml').read_text()
    assert mount[-1]=='/home/agent/.codex/rbs.config.toml'
    assert profile.isolated_environment()=={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','RBS_FIXTURE_KEY':'fixture-only'}


def test_missing_profile_fails_without_fallback(profile_home):
    (profile_home/'rbs.config.toml').unlink()
    with pytest.raises(RuntimeError,match='profile missing'):
        profile.profile_settings()


def test_missing_key_names_only_variable(profile_home,monkeypatch):
    monkeypatch.delenv('RBS_FIXTURE_KEY')
    with pytest.raises(RuntimeError,match='RBS_FIXTURE_KEY'):
        profile.isolated_environment()
