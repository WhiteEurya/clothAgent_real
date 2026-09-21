import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from cloth_agent.claude_molmo_view import MolmoOrientationError, prepare_molmo_view
from cloth_agent.image_tools_mcp import ImageTools, audit, main, orientation_guard
from cloth_agent.planner_backend import RemoteClaudeBackend


UNCERTAIN = dict(status='UNCERTAIN', image_id=None, collar_pixel_xy=None,
                 hem_pixel_xy=None, failure_reason='IMAGE_UNAVAILABLE',
                 reason='Read returned empty output; no image was visible.')


def hook_payload(candidate=UNCERTAIN, event='PreToolUse'):
    return dict(hook_event_name=event, tool_name='StructuredOutput', tool_input=candidate,
                last_assistant_message=json.dumps(candidate), stop_hook_active=False)


@pytest.fixture
def inspected(tmp_path):
    Image.new('RGB', (120, 80), 'white').save(tmp_path / 'image_0.png')
    main(['--job', str(tmp_path), '--image-count', '1', '--edit-limit', '6',
          '--orientation-correction', '--prepare'])
    tools = ImageTools(tmp_path, 1, edit_limit=6)
    crop = tools.call('crop_image', {'image_id': 'image_0', 'box': [10, 10, 90, 60]})
    for i, view in enumerate((tools.views['image_0'], crop)):
        for status in ('started', 'completed'):
            audit(tmp_path, dict(kind='read', tool='Read', status=status, tool_use_id=f'read-{i}',
                                 arguments={'file_path': view['path']}))
    return tools


def latest(tools):
    return json.loads((tools.job / 'image_tool_calls.jsonl').read_text().splitlines()[-1])


@pytest.mark.parametrize('event', ['PreToolUse', 'Stop'])
def test_missing_pixels_are_audited_without_correction_or_budget_reset(inspected, event):
    tools = inspected
    before = tools.edit_budget()
    response = orientation_guard(tools.job, hook_payload(event=event))
    assert response == {}
    check = latest(tools)
    assert check['status'] == 'not_requested'
    assert check['classification'] == 'IMAGE_UNAVAILABLE'
    assert check['candidate'] == UNCERTAIN
    assert tools.edit_budget() == before
    assert orientation_guard(tools.job, hook_payload()) == {}
    assert orientation_guard(tools.job, hook_payload(event='Stop')) == {}
    assert not (tools.job / 'orientation_correction.json').exists()
    restarted = ImageTools(tools.job, 1, edit_limit=6)
    assert len(restarted.views) == 2
    rotated = restarted.call('rotate_image', {'image_id': 'image_0', 'degrees_clockwise': 90})
    assert rotated['edit_budget']['used'] == 2
    assert len(restarted.views) == 3
    assert orientation_guard(tools.job, hook_payload()) == {}


@pytest.mark.parametrize('case', ['ambiguous', 'ready', 'malformed', 'budget', 'error',
    'read_error', 'pending', 'missing_audit', 'truncated_audit', 'no_reads', 'active_hook'])
def test_audit_conditions_never_override_a_missing_image_report(inspected, case):
    tools = inspected
    request = hook_payload()
    if case == 'ambiguous':
        request = hook_payload({**UNCERTAIN, 'failure_reason': 'VISUAL_AMBIGUITY',
                                'reason': 'The collar is hidden under fabric.'})
    elif case == 'ready':
        request = hook_payload({**UNCERTAIN, 'status': 'READY', 'failure_reason': None})
    elif case == 'malformed':
        request = hook_payload({'status': 'UNCERTAIN', 'reason': UNCERTAIN['reason']})
    elif case == 'budget':
        for i in range(5):
            tools.call('rotate_image', {'image_id': 'image_0', 'degrees_clockwise': i * 15})
    elif case == 'error':
        audit(tools.job, dict(tool='rotate_image', status='error', error='failure'))
    elif case == 'read_error':
        audit(tools.job, dict(kind='read', tool='Read', status='failed', tool_use_id='bad-read'))
    elif case == 'pending':
        audit(tools.job, dict(kind='tool_lifecycle', tool='mcp__cloth_image__rotate_image',
                              status='started', tool_use_id='pending'))
    elif case == 'active_hook':
        request['stop_hook_active'] = True
    else:
        path = tools.job / 'image_tool_calls.jsonl'
        if case == 'missing_audit':
            path.unlink()
        elif case == 'truncated_audit':
            with path.open('a') as f:
                f.write('{"unfinished":\n')
        else:
            rows = [row for row in path.read_text().splitlines() if json.loads(row).get('kind') != 'read']
            path.write_text('\n'.join(rows) + '\n')
    assert orientation_guard(tools.job, request) == {}
    assert latest(tools)['classification'] == {
        'ambiguous': 'VISUAL_AMBIGUITY', 'ready': 'READY', 'malformed': 'UNCLASSIFIED',
    }.get(case, 'IMAGE_UNAVAILABLE')
    assert latest(tools)['status'] == 'not_requested'


def test_backend_only_registers_guard_for_orientation(inspected, monkeypatch):
    from cloth_agent.planner_backend import BackendResult
    backend = RemoteClaudeBackend()
    setups = []
    def invoke(**kwargs):
        setups.append(backend._image_tool_setup('/tmp/cloth_remote_test', 1))
        return BackendResult('{}', '', 0, ())
    monkeypatch.setattr(backend, '_invoke', invoke)
    request = dict(prompt='test', image_paths=[inspected.job / 'image_0.png'], schema={}, system_prompt='test')
    backend.invoke(**request, image_edit_limit=6, orientation_correction=True)
    backend.invoke(**request, image_edit_limit=2)
    assert '--orientation-correction' in setups[0][0]
    assert '--orientation-correction' not in setups[1][0]
    with pytest.raises(ValueError, match='finite edit budget'):
        backend.invoke(**request, orientation_correction=True)


@pytest.mark.parametrize('reason', [
    'Read succeeded but I failed to identify the collar.',
    'Image tools worked; the collar is ambiguous and visual orientation failed.',
    '读取工具成功，但没有看清衣领，无法验证方向。',
])
def test_visual_failure_is_not_misread_as_tool_failure(inspected, reason):
    assert orientation_guard(inspected.job, hook_payload({**UNCERTAIN, 'reason': reason,
                             'failure_reason': 'VISUAL_AMBIGUITY'})) == {}
    assert latest(inspected)['classification'] == 'VISUAL_AMBIGUITY'


@pytest.mark.parametrize('reason', [
    'Only two views were successfully Read before the image tools stopped returning results.',
    'Every subsequent rotate_image, list_images and Read call returned no result.',
    'The rotate_image tool failed to return a result.',
    '图像工具没有返回结果，无法验证旋转后的方向。',
])
def test_reported_missing_tool_results_are_not_regex_corrected(inspected, reason):
    assert orientation_guard(inspected.job, hook_payload({**UNCERTAIN, 'reason': reason})) == {}
    assert latest(inspected)['classification'] == 'IMAGE_UNAVAILABLE'
    assert latest(inspected)['status'] == 'not_requested'


# Real shell/bootstrap, MCP, hooks and replay; just the model/transport are fake.
STUB = '''
import json, os, re, shlex, subprocess, sys
from pathlib import Path
settings = json.loads(Path(sys.argv[sys.argv.index('--settings') + 1]).read_text())
config = json.loads(Path(sys.argv[sys.argv.index('--mcp-config') + 1]).read_text())
server = config['mcpServers']['cloth_image']
job = Path.cwd()
assert sys.argv[sys.argv.index('--max-turns') + 1] == '16'
assert '--no-session-persistence' in sys.argv
assert (job / 'orientation_correction.json').exists() is False
def hook(event, tool='', args=None, identity='test', response=None):
    result = {}
    payload = dict(hook_event_name=event, tool_name=tool, tool_use_id=identity,
                   tool_input=args or {}, last_assistant_message=json.dumps(args or {}),
                   stop_hook_active=False, tool_response=response)
    for entry in settings['hooks'][event]:
        if re.fullmatch(entry.get('matcher', '.*'), tool):
            for spec in entry['hooks']:
                out = subprocess.run(shlex.split(spec['command']), input=json.dumps(payload),
                                     capture_output=True, text=True, check=True)
                if out.stdout.strip():
                    result.update(json.loads(out.stdout))
    return result
def call(name, args, identity):
    tool = 'mcp__cloth_image__' + name
    print(json.dumps({'type':'assistant','message':{'content':[
        {'type':'tool_use','id':identity,'name':tool,'input':args}]}}), flush=True)
    hook('PreToolUse', tool, args, identity)
    out = subprocess.run([server['command'], *server['args']],
        input=json.dumps({'jsonrpc':'2.0', 'id':1, 'method':'tools/call',
                          'params':{'name':name, 'arguments':args}})+'\\n',
        capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)['result']
    assert not result.get('isError'), result
    assert result['content'][1]['type'] == 'image'
    hook('PostToolUse', tool, args, identity, result)
    print(json.dumps({'type':'user','message':{'content':[
        {'type':'tool_result','tool_use_id':identity,'content':result['content']}]}}), flush=True)
    return json.loads(result['content'][0]['text'])
call('view_image', {'image_id':'image_0'}, 'view-original')
crop = call('crop_image', {'image_id':'image_0','box':[10,10,90,60]}, 'crop')
uncertain = dict(status='UNCERTAIN', image_id=None, collar_pixel_xy=None, hem_pixel_xy=None,
                 failure_reason='IMAGE_UNAVAILABLE', reason='No visual content was visible.')
assert Path(crop['path']).is_file()
if os.environ['ORIENTATION_STUB_OUTCOME'] == 'ready':
    rotated = call('rotate_image', {'image_id':'image_0','degrees_clockwise':90}, 'rotate')
    assert rotated['edit_budget']['used'] == 2
    answer = dict(status='READY', image_id=rotated['image_id'], collar_pixel_xy=[40,20],
                  hem_pixel_xy=[40,95], failure_reason=None, reason='Collar above hem.')
else:
    answer = uncertain
assert hook('PreToolUse', 'StructuredOutput', answer) == {}
assert hook('Stop', args=answer) == {}
print(json.dumps({'type':'result','structured_output':answer,'is_error':False,'num_turns':8}), flush=True)
'''


@pytest.mark.parametrize('outcome', ['ready', 'uncertain'])
def test_full_orientation_bridge_direct_image_results_without_extra_reads(tmp_path, monkeypatch, outcome):
    image = tmp_path / 'camera_A_rgb_upright.png'
    Image.new('RGB', (120, 80), 'white').save(image)
    stub = tmp_path / 'stub.py'
    stub.write_text(STUB)
    real_run, real_popen = subprocess.run, subprocess.Popen
    starts, uploads = [], []
    monkeypatch.setenv('CLOTH_REMOTE_IMAGE_PYTHON', sys.executable)
    monkeypatch.setenv('ORIENTATION_STUB_OUTCOME', outcome)
    def substitute(command):
        remote = command[-1]
        if 'claude -p' in remote:
            starts.append(command)
            remote = remote.replace('claude -p', f'{shlex.quote(sys.executable)} {shlex.quote(str(stub))} -p')
            remote = re.sub(r'curl -fsSL --http1.1 --connect-timeout 20 --max-time 120 \S+ -o (\S+)',
                            lambda m: f'cp {shlex.quote(str(image))} {m[1]}', remote)
        return ['sh', '-c', remote]
    def run(command, **kwargs):
        return real_run(substitute(command) if command[0] == 'ssh' else command, **kwargs)
    def popen(command, **kwargs):
        return real_popen(substitute(command) if command[0] == 'ssh' else command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    backend = RemoteClaudeBackend()
    def upload(path):
        uploads.append(path)
        return 'https://example.invalid/image'
    monkeypatch.setattr(backend, '_upload', upload)
    output = tmp_path / 'orientation'
    if outcome == 'uncertain':
        with pytest.raises(MolmoOrientationError, match='could not establish'):
            prepare_molmo_view(backend, image, output, timeout_s=30)
    else:
        prepare_molmo_view(backend, image, output, timeout_s=30)
    report = json.loads((output / 'selection.json').read_text())
    assert len(starts) == len(uploads) == 1
    assert report['correction_checks']
    assert all(e['status'] == 'not_requested' for e in report['correction_checks'])
    assert report['correction_applied'] is False
    debug = Path(report['image_debug_directory'])
    assert 'Host orientation audit' in (debug / 'claude_transcript.md').read_text()
    assert json.loads((debug / 'request.json').read_text())['overall_timeout_s'] == 30
    assert json.loads((debug / 'image_debug.json').read_text())['audit_complete']
    assert (output / 'molmo_input').exists() == (outcome == 'ready')
    if outcome == 'ready':
        assert report['status'] == 'READY'
        assert len(report['image_sources']) == 3
        assert all(v['verification'] == 'VERIFIED' and v['image_delivery_status'] == 'VERIFIED'
                   and v['read_status'] == 'NO_READ_RECORDED'
                   for v in report['image_sources'])
    else:
        assert report['status'] == 'FAILED_NO_MOLMO'
        assert report['failure_reason'] == 'IMAGE_UNAVAILABLE'
        assert report['failure_reason_source'] == 'model_report'
