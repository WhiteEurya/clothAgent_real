import json
import subprocess
import sys
from pathlib import Path
from cloth_agent.remote_output import compact

WRAPPER = Path(__file__).resolve().parents[1] / 'cloth_agent/remote_output.py'


def test_spool_and_compact(tmp_path):
    image = {'type': 'image', 'source': {'type': 'base64', 'data': 'a' * 1000000}}
    event = {'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 'test', 'content': [image]}]},
        'tool_use_result': [image]}
    terminal = {'type': 'result', 'subtype': 'success', 'result': {'answer': 42}}
    code = 'import json; print(json.dumps(' + repr(event) + ')); print(json.dumps(' + repr(terminal) + '))'
    script = tmp_path / 'model.py'; script.write_text(code)
    r = subprocess.run([sys.executable, str(WRAPPER), sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0
    assert 1000000 < len(r.stdout) < 1001000
    received = json.loads(r.stdout.splitlines()[0])
    assert received['message'] == event['message']
    assert 'tool_use_result' not in received
    assert json.loads(r.stdout.splitlines()[-1]) == terminal
    assert (tmp_path / 'claude_raw.jsonl').stat().st_size > 1000000


def test_truncation_is_remote_error_with_preserved_raw(tmp_path):
    job = tmp_path / 'job'; job.mkdir()
    r = subprocess.run([sys.executable, str(WRAPPER), sys.executable, '-c', 'print(\'{"type":\', end="")'], cwd=job, capture_output=True, text=True)
    assert r.returncode == 65
    assert 'REMOTE_OUTPUT_INVALID' in r.stderr
    assert not r.stdout
    assert (tmp_path / 'job.failed.jsonl').read_text() == '{"type":'


def test_unique_evidence_and_terminal_unchanged():
    event = {'type': 'user', 'message': {'content': []}, 'tool_use_result': ['unique']}
    assert compact(event) == event
    terminal = dict(event, type='result')
    assert compact(terminal) == terminal


def test_delivery_evidence_survives_compaction():
    import base64
    import io
    from PIL import Image
    from cloth_agent.image_tools_mcp import image_content_summary
    data = io.BytesIO()
    Image.new('RGB', (4, 4), 'red').save(data, format='PNG')
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                 'data': base64.b64encode(data.getvalue()).decode()}}]
    event = {'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 't1', 'content': content}]}, 'tool_use_result': content}
    retained = compact(event)['message']['content'][0]['content']
    summary = image_content_summary(retained)
    assert summary == image_content_summary(content)
    assert summary['image_count'] == 1
    assert summary['status'] == 'VALID_IMAGE'


def test_context_files_do_not_enter_initial_model_prompt(tmp_path):
    envelope = {'prompt': 'Read context/manifest.json as needed.', 'files': {
        'manifest.json': '{"files":["00.json"]}', '00.json': '{"history":"large evidence"}'}}
    script = tmp_path / 'model.py'
    script.write_text("import sys,json\nfrom pathlib import Path\np=sys.stdin.read()\nassert 'large evidence' not in p\nassert json.loads(Path('context/00.json').read_text())['history']=='large evidence'\nprint(json.dumps({'type':'result','result':'ok'}))")
    result = subprocess.run([sys.executable, str(WRAPPER), '--context-envelope', sys.executable, str(script)],
        input=json.dumps(envelope), cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['result'] == 'ok'


def test_opt_in_timing_preserves_input_terminal_and_remote_intervals(tmp_path):
    script = tmp_path / 'model.py'
    script.write_text("import sys,json,time\nassert sys.stdin.read()=='original prompt'\n"
                      "print(json.dumps({'type':'system','subtype':'init'}),flush=True)\n"
                      "time.sleep(0.12)\n"
                      "print(json.dumps({'type':'assistant','message':{'content':[]}}),flush=True)\n"
                      "print(json.dumps({'type':'result','result':'ok'}),flush=True)\n")
    result = subprocess.run([sys.executable,str(WRAPPER),'--record-event-timing','--context-envelope',
                             sys.executable,str(script)],cwd=tmp_path,capture_output=True,text=True,
                            input=json.dumps({'prompt':'original prompt','files':{}}))
    assert result.returncode == 0,result.stderr
    rows=[json.loads(s) for s in result.stdout.splitlines()]
    assert rows[1]['_cloth_timing']['elapsed_s']-rows[0]['_cloth_timing']['elapsed_s'] >= .10
    assert rows[-1] == {'type':'result','result':'ok'}
    assert '__CLOTH_PROFILE__' in result.stderr
    assert '_cloth_timing' not in (tmp_path/'claude_raw.jsonl').read_text()


def test_event_arrives_before_cli_finishes(tmp_path):
    import selectors
    script=tmp_path/'model.py'
    script.write_text("import json,time\nfrom pathlib import Path\n"
                      "print(json.dumps({'type':'system','subtype':'api_retry','error_status':524,'attempt':1}),flush=True)\n"
                      "while not Path('release').exists(): time.sleep(.02)\n"
                      "print(json.dumps({'type':'result','result':'ok'}),flush=True)\n")
    process=subprocess.Popen([sys.executable,str(WRAPPER),'--record-event-timing',sys.executable,str(script)],
                             cwd=tmp_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            assert selector.select(timeout=3), 'Event was buffered until process exit'
        event=json.loads(process.stdout.readline())
        assert event['subtype']=='api_retry' and event['error_status']==524
        assert event['_cloth_timing']['elapsed_s']>=0
        assert process.poll() is None
        (tmp_path/'release').touch()
        stdout,stderr=process.communicate(timeout=3)
        assert process.returncode==0
        assert json.loads(stdout)['type']=='result'
    finally:
        if process.poll() is None:
            (tmp_path/'release').touch()
            process.communicate(timeout=3)


def test_remote_timeout_keeps_partial_events_without_fake_result(tmp_path):
    import shutil
    import pytest
    timeout=shutil.which('timeout')
    if not timeout: pytest.skip('GNU timeout unavailable')
    job=tmp_path/'job';job.mkdir()
    code="import json,time; print(json.dumps({'type':'system','subtype':'api_retry','error_status':524}),flush=True); time.sleep(20)"
    result=subprocess.run([sys.executable,str(WRAPPER),'--record-event-timing',timeout,'--kill-after=1s','.3s',sys.executable,'-c',code],
                          cwd=job,capture_output=True,text=True,timeout=4)
    assert result.returncode==124
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(events)==1 and events[0]['subtype']=='api_retry'
    assert not any(e['type']=='result' for e in events)
    assert '"terminal_received":false' in result.stderr
    assert (tmp_path/'job.failed.jsonl').is_file()


def test_failing_process_cannot_publish_success_terminal(tmp_path):
    code="import json,sys; print(json.dumps({'type':'result','result':'ok'})); sys.exit(1)"
    result=subprocess.run([sys.executable,str(WRAPPER),sys.executable,'-c',code],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==1
    assert not result.stdout
    assert 'unaccepted_terminal' in result.stderr


def test_silent_cli_heartbeat_identifies_waiting(tmp_path):
    code="import json,time; time.sleep(5.2); print(json.dumps({'type':'result','result':'ok'}))"
    result=subprocess.run([sys.executable,str(WRAPPER),sys.executable,'-c',code],cwd=tmp_path,capture_output=True,text=True,timeout=8)
    events=[json.loads(line.split(' ',1)[1]) for line in result.stderr.splitlines() if line.startswith('__CLOTH_PROGRESS__ ')]
    pulse=next(e for e in events if e['phase']=='waiting_for_cli')
    assert pulse['event_count']==0 and pulse['idle_s']>=5
    assert result.returncode==0


def test_text_only_stream_has_no_images(tmp_path):
    code="import json,sys; m=json.loads(sys.stdin.readline()); assert m['message']['content']==[{'type':'text','text':'saved evidence'}]; print(json.dumps({'type':'result','subtype':'success','result':'ok'}))"
    result=subprocess.run([sys.executable,str(WRAPPER),'--text-only',sys.executable,'-c',code],
                          input='saved evidence',text=True,capture_output=True,cwd=tmp_path)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['result']=='ok'
