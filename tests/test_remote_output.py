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
