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


def test_profile_and_replay_artifacts_under_ascii_locale(profile_home, tmp_path):
    import os
    import subprocess
    import sys
    base = profile_home / 'config.toml'
    base.write_text(base.read_text().replace('name="fixture"', 'name="中文配置"'), encoding='utf-8')
    work = tmp_path / 'ascii_work'
    work.mkdir()
    code = '''
import locale
from pathlib import Path
import sys
from cloth_agent.codex_profile import isolated_profile_mount, profile_settings
from cloth_agent.experiments.discovery_data import save, read
assert locale.getpreferredencoding(False).lower() in ('ascii', 'ansi_x3.4-1968')
work = Path(sys.argv[1])
isolated_profile_mount(work)
name = profile_settings()['model_providers']['rbs_test']['name']
assert len(name) == 4
assert name in (work / 'rbs.config.toml').read_text(encoding='utf-8')
save(work / 'result.json', {'name': name})
assert read(work / 'result.json') == {'name': name}
'''
    result = subprocess.run([sys.executable, '-c', code, str(work)],
        env=dict(os.environ, LC_ALL='C', LANG='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0'),
        capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
