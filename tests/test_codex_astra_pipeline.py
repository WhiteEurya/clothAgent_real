import json
from pathlib import Path

import pytest

from cloth_agent.codex_astra_pipeline import build_prompt, run


@pytest.fixture
def fake_codex(tmp_path):
    p=tmp_path/'codex'
    p.write_text('''#!/usr/bin/env python3
import json,sys
from pathlib import Path
prompt=sys.stdin.read()
assert prompt
assert sys.argv[sys.argv.index('--model')+1]=='gpt-6-astra'
assert 'model_reasoning_effort="high"' in sys.argv
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('motion.move_joints(left=[0,0,0,0,0,0])')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}))
''')
    p.chmod(0o755)
    return str(p)


def test_local_exec_records_usage_without_claiming_task_success(tmp_path,fake_codex):
    out=tmp_path/'run'
    result=run('Fold this shirt',workspace=tmp_path,output=out,binary=fake_codex)
    assert result['status']=='CODE_READY'
    assert result['task_success'] is None
    assert result['usage']=={'input_tokens':100,'output_tokens':20}
    request=json.loads((out/'request.json').read_text())
    assert '--approve-for-me' in request['command']
    assert request['reasoning_effort']=='high'
    assert (out/'answer.txt').read_text()=='motion.move_joints(left=[0,0,0,0,0,0])'


def test_dry_run_does_not_launch_and_read_only_is_explicit(tmp_path):
    result=run('Inspect',workspace=tmp_path,output=tmp_path/'dry',binary='/bin/false',
               read_only=True,dry_run=True)
    assert result['status']=='DRY_RUN'
    request=json.loads((tmp_path/'dry/request.json').read_text())
    assert request['command'][request['command'].index('--sandbox')+1]=='read-only'
    assert '不连接或操作机器人' in request['prompt']
    assert not (tmp_path/'dry/events.jsonl').exists()


def test_failure_is_not_success(tmp_path):
    result=run('Inspect',workspace=tmp_path,output=tmp_path/'fail',binary='/bin/false')
    assert result['status']=='FAILED'
    assert result['returncode']==1


def test_timeout_reaps_child(tmp_path):
    binary=tmp_path/'slow'
    binary.write_text('#!/bin/sh\nsleep 30\n')
    binary.chmod(0o755)
    result=run('Inspect',workspace=tmp_path,output=tmp_path/'timeout',binary=str(binary),timeout_s=.1)
    assert result['status']=='TIMEOUT'
    assert result['duration_s']<5


def test_empty_task_rejected():
    with pytest.raises(ValueError):
        build_prompt('  ')


def test_pipeline_hands_generated_code_to_planner(tmp_path,fake_codex,monkeypatch):
    from cloth_agent import codex_motion
    scene=tmp_path/'scene.json';scene.write_text('{}')
    state=tmp_path/'state.json';state.write_text('{}')
    captured=[]
    def handoff(code,scene_path,state_path,output):
        captured.append((code.read_text(),scene_path,state_path))
        output.write_text('{"success":true}')
        return {'success':True}
    monkeypatch.setattr(codex_motion,'handoff',handoff)
    result=run('Fold',workspace=tmp_path,output=tmp_path/'planned',binary=fake_codex,scene=scene,state=state)
    assert result['status']=='PLANNED' and result['physical_execution'] is False
    assert captured==[('motion.move_joints(left=[0,0,0,0,0,0])',scene,state)]
