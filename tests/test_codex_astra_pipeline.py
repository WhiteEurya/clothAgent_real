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
print(json.dumps({'type':'thread.started','thread_id':'test-session'}))
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


def test_resume_uses_explicit_id_and_preserves_read_only_and_images(tmp_path,fake_codex):
    from cloth_agent.codex_astra_pipeline import build_command
    image=tmp_path/'image.png';image.write_bytes(b'image')
    first=run('first',workspace=tmp_path,output=tmp_path/'first',binary=fake_codex,read_only=True)
    assert first['session_id']=='test-session'
    second=run('next',workspace=tmp_path,output=tmp_path/'next',binary=fake_codex,
               read_only=True,session_id=first['session_id'],images=[image])
    assert second['status']=='AGENT_FINISHED' and second['session_id']==first['session_id']
    command=json.loads((tmp_path/'next/request.json').read_text())['command']
    assert command[command.index('resume')+1]=='test-session'
    assert '--last' not in command and '--image' in command
    assert command.index('--sandbox')<command.index('resume')
    assert command[command.index('--sandbox')+1]=='read-only'
    assert command[-1]=='-'


def test_resumed_thread_mismatch_is_failure(tmp_path,fake_codex):
    result=run('next',workspace=tmp_path,output=tmp_path/'bad',binary=fake_codex,
               read_only=True,session_id='other-thread')
    assert result['status']=='FAILED' and 'mismatch' in result['error']


@pytest.mark.parametrize('case,expected', [
    ('recovered', 'AGENT_FINISHED'), ('unrecovered', 'FAILED'),
    ('explicit_failure', 'FAILED'), ('unknown_error', 'FAILED'),
    ('error_after_complete', 'FAILED'), ('new_incomplete_turn', 'FAILED'),
    ('empty_answer', 'FAILED'), ('no_completion', 'FAILED'),
])
def test_stream_recovery_requires_final_completion_and_answer(tmp_path, fake_codex, case, expected):
    p=Path(fake_codex); source=p.read_text()
    completion="print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}))"
    reconnect="print(json.dumps({'type':'error','message':'Reconnecting... 5/5 (stream disconnected before completion: websocket closed by server before response.completed)'}))"
    if case=='recovered': replacement=reconnect+'\n'+completion
    elif case=='unrecovered': replacement=reconnect
    elif case=='explicit_failure': replacement="print(json.dumps({'type':'turn.failed','error':{'message':'failed'}}))\n"+completion
    elif case=='unknown_error': replacement="print(json.dumps({'type':'error','message':'authentication failed'}))\n"+completion
    elif case=='error_after_complete': replacement=completion+'\n'+reconnect
    elif case=='new_incomplete_turn': replacement=completion+"\nprint(json.dumps({'type':'turn.started'}))"
    elif case=='no_completion': replacement=''
    else:
        source=source.replace(".write_text('motion.move_joints(left=[0,0,0,0,0,0])')", ".write_text('   ')")
        replacement=completion
    p.write_text(source.replace(completion,replacement))
    result=run('fold',workspace=tmp_path,output=tmp_path/'out',binary=fake_codex,read_only=True)
    assert result['status']==expected
    if case=='recovered':
        assert result['errors']==[] and len(result['recovered_errors'])==1
        assert result['final_turn_completed'] and result['completed_turns']==1
