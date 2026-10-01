from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from cloth_agent.token_usage import (
    format_summary, main, parse_usage, record_usage, summarize_usage, tracked_call,
)


def envelope(**overrides):
    return json.dumps({
        "type": "result", "subtype": "success", "session_id": "same-session",
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 300, "cache_creation_input_tokens": 40},
        "total_cost_usd": 0.025,
        "modelUsage": {"claude-test": {"inputTokens": 100, "outputTokens": 20,
                       "cacheReadInputTokens": 300, "cacheCreationInputTokens": 40,
                       "costUSD": 0.025}},
        **overrides,
    })


def test_json_and_stream_count_terminal_only():
    stream = '\n'.join([
        json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 9000}}}),
        envelope(),
    ])
    assert parse_usage(stream) == parse_usage(envelope())
    assert parse_usage(stream)["total_tokens"] == 460
    assert parse_usage(stream)["total_cost_usd"] == 0.025


def test_model_only_fallback_and_no_double_counting():
    assert parse_usage(envelope(usage=None))["total_tokens"] == 460
    assert parse_usage(envelope())["input_tokens"] == 100
    data = json.loads(envelope())
    data['modelUsage']['second-model'] = data['modelUsage']['claude-test'].copy()
    assert parse_usage(json.dumps(data))["input_tokens"] == 100
    data.pop('usage')
    assert parse_usage(json.dumps(data))["input_tokens"] == 200


@pytest.mark.parametrize('stdout', ['', 'not json', '[]', '{"type":"assistant","usage":{"input_tokens":8}}'])
def test_missing_usage_is_unknown(stdout):
    usage = parse_usage(stdout)
    assert usage['reported'] is False
    assert usage['input_tokens'] is None
    assert usage['total_tokens'] is None
    assert usage['total_cost_usd'] is None


def test_invalid_and_partial_counters_are_not_silently_zero():
    usage = parse_usage(envelope(modelUsage={}, total_cost_usd=float('nan'), usage={
        'input_tokens': -1, 'output_tokens': True, 'cache_read_input_tokens': 3.5,
        'cache_creation_input_tokens': 12,
    }))
    assert usage['input_tokens'] is None
    assert usage['output_tokens'] is None
    assert usage['cache_read_input_tokens'] is None
    assert usage['cache_creation_input_tokens'] == 12
    assert usage['total_tokens'] is None
    assert usage['total_cost_usd'] is None


def test_retries_same_session_and_restart_are_separate_calls(tmp_path):
    (tmp_path / 'run_metadata.json').write_text('{}')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for status in ['failed', 'completed']:
        record_usage(workspace, stage='plan', backend='local', stdout=envelope(), status=status)
    record_usage(tmp_path / 'results' / 'debug', stage='evaluate', backend='remote', stdout=envelope())
    summary = summarize_usage(tmp_path)
    assert summary['totals']['calls'] == 3
    assert summary['totals']['total_tokens'] == 1380
    assert summary['totals']['total_cost_usd'] == pytest.approx(.075)
    assert summary['by_stage']['plan']['calls'] == 2
    assert summary['by_model']['claude-test']['total_tokens'] == 1380
    rows = [json.loads(s) for s in (tmp_path / 'results/token_usage.jsonl').read_text().splitlines()]
    assert len({r['call_id'] for r in rows}) == 3


def test_failed_process_recorded_before_caller_validation(tmp_path):
    completed = subprocess.CompletedProcess(['claude'], 1, envelope(is_error=True), 'error')
    assert tracked_call(lambda: completed, usage_run_dir=tmp_path, usage_stage='plan') is completed
    row = json.loads((tmp_path / 'results/token_usage.jsonl').read_text())
    assert row['status'] == 'failed'
    assert row['total_tokens'] == 460


def test_timeout_bytes_and_unknown_call_are_visible(tmp_path):
    def timeout():
        raise subprocess.TimeoutExpired('claude', 1, output=envelope().encode())
    with pytest.raises(subprocess.TimeoutExpired):
        tracked_call(timeout, usage_run_dir=tmp_path, usage_stage='plan')
    def fail():
        raise OSError('missing binary')
    with pytest.raises(OSError):
        tracked_call(fail, usage_run_dir=tmp_path, usage_stage='plan')
    summary = summarize_usage(tmp_path)
    assert summary['totals']['calls'] == 2
    assert summary['totals']['total_tokens'] == 460
    assert summary['totals']['calls_without_usage'] == 1
    assert summary['totals']['missing_fields']['total_tokens'] == 1
    assert 'known subtotal only' in format_summary(summary)


def test_concurrent_writers(tmp_path):
    def write(_):
        record_usage(tmp_path, stage='plan', backend='local', stdout=envelope())
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(40)))
    summary = summarize_usage(tmp_path)
    assert summary['totals']['calls'] == 40
    assert summary['totals']['total_tokens'] == 18400
    assert summary['invalid_records'] == 0


def test_usage_write_failure_does_not_change_result(tmp_path, caplog):
    (tmp_path / 'results').write_text('cannot create a directory here')
    result = subprocess.CompletedProcess(['claude'], 0, envelope(), '')
    assert tracked_call(lambda: result, usage_run_dir=tmp_path, usage_stage='plan') is result
    assert 'Could not save Claude token usage' in caplog.text


def test_cli_empty_and_truncated_ledger(tmp_path, capsys):
    assert main(['--run-dir', str(tmp_path), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['totals']['calls'] == 0
    record_usage(tmp_path, stage='plan', backend='local', stdout=envelope())
    with (tmp_path / 'results/token_usage.jsonl').open('a') as stream:
        stream.write('{"schema_ver')
    assert main(['--run-dir', str(tmp_path), '--json']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['invalid_records'] == 1
    assert summary['totals']['total_tokens'] == 460


def test_local_client_integration(tmp_path, monkeypatch):
    from cloth_agent.claude import ClaudeCodeClient
    monkeypatch.setattr('cloth_agent.claude.shutil.which', lambda _: '/fake/claude')
    monkeypatch.setattr('cloth_agent.claude.subprocess.run', lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 0, envelope(), ''))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    ClaudeCodeClient().invoke('generate a plan', workspace)
    assert summarize_usage(tmp_path)['by_stage']['code_generation']['total_tokens'] == 460


def test_remote_backend_integration(tmp_path, monkeypatch):
    from PIL import Image
    from cloth_agent.planner_backend import RemoteClaudeBackend, PlannerBackendError
    image = tmp_path / 'rgb.png'
    Image.new('RGB', (2, 2)).save(image)
    backend = RemoteClaudeBackend(image_tools=False)
    monkeypatch.setattr(backend, '_upload', lambda _: 'https://example.test/rgb.png')
    def run(command, **kwargs):
        cleanup = 'rm -rf --' in command[-1] and 'claude' not in command[-1]
        return subprocess.CompletedProcess(command, 0 if cleanup else 1,
                                           '' if cleanup else envelope(is_error=True), '')
    monkeypatch.setattr('cloth_agent.planner_backend.subprocess.run', run)
    with pytest.raises(PlannerBackendError):
        backend.invoke(prompt='plan', image_paths=[image], schema={}, system_prompt='plan',
                       usage_run_dir=tmp_path, usage_stage='visual_planning')
    totals = summarize_usage(tmp_path)['totals']
    assert totals['calls'] == 1  # SSH cleanup is not a model invocation.
    assert totals['total_tokens'] == 460


def test_remote_stream_integration(tmp_path, monkeypatch):
    from PIL import Image
    from cloth_agent.planner_backend import RemoteClaudeBackend
    image = tmp_path / 'rgb.png'
    Image.new('RGB', (2, 2)).save(image)
    (tmp_path / 'run_metadata.json').write_text('{}')
    backend = RemoteClaudeBackend(image_tools=False)
    monkeypatch.setattr(backend, '_upload', lambda _: 'https://example.test/rgb.png')
    stream = json.dumps({'type': 'assistant', 'message': {'usage': {'input_tokens': 9000}}}) + '\n' + envelope()
    monkeypatch.setattr(backend, '_run_streaming', lambda command, *a:
                        subprocess.CompletedProcess(command, 0, stream, ''))
    monkeypatch.setattr('cloth_agent.planner_backend.subprocess.run', lambda command, **kw:
                        subprocess.CompletedProcess(command, 0, '', ''))
    backend.invoke(prompt='plan', image_paths=[image], schema={}, system_prompt='plan',
                   debug_dir=tmp_path / 'results/debug', usage_stage='visual_planning')
    summary = summarize_usage(tmp_path)
    assert summary['totals']['calls'] == 1
    assert summary['totals']['total_tokens'] == 460


def test_cli_subcommand(tmp_path, capsys):
    from cloth_agent.cli import main as cli_main
    record_usage(tmp_path, stage='plan', backend='local', stdout=envelope())
    assert cli_main(['token-usage', '--run-dir', str(tmp_path), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['totals']['total_tokens'] == 460
