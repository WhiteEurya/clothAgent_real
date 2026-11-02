import json
from pathlib import Path
import tempfile
import pytest
from rax_cloth.official import smoke

HERE=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode',['mock','replay'])
def test_full_official_components_and_resume(mode,monkeypatch):
    monkeypatch.setenv('RAX_ARTIFACT_ROOT','/tmp')
    monkeypatch.setenv('RAX_SKILL_LIBRARY','/tmp/unused')
    with tempfile.TemporaryDirectory(prefix='rax-full-') as tmp:
        root=Path(tmp)/'session'
        manifest=HERE/'configs/replay.example.json' if mode=='replay' else None
        result=smoke(root,mode,manifest)
        assert result['interface_test']=='passed'
        assert result['physical_actions']==0 and result['task_success'] is None
        assert result['official_commit']=='36c926ca3cf15921c6fc746a4bf309c26b0aa3c4'
        ws=root/'agent-workspace'
        assert (ws/'skill-library/.git').is_dir()
        assert not list((ws/'skill-library/skills').glob('*/SKILL.md'))
        for skill in ['autonomous-control','experience-memory','continual-learning','manual-robot-control']:
            assert (ws/'.agents/skills'/skill/'SKILL.md').is_file()
        agents=(ws/'AGENTS.md').read_text()
        assert 'Offline mode override' in agents and 'rby1_' not in agents
        contract=json.loads((ws/'MCP_CONTRACT.json').read_text())
        assert 'skill_library_merge' not in contract['tools']
        assert not list(ws.rglob('*xarm*'))
        old=(root/'mcp.jsonl').read_text()
        assert 'mcp_request' in old and 'mcp_result' in old
        assert smoke(root,mode,manifest)['interface_test']=='passed'
        assert len((root/'mcp.jsonl').read_text())>len(old)
        assert len(list((root/'observations').glob('[0-9]*')))==2
