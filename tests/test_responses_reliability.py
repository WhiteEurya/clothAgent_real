import json
from types import SimpleNamespace

from scripts import responses_reliability_test as check


def test_reliability_report_separates_recovery_from_first_attempt_and_failure(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        directory = check.Path(command[command.index('--output-dir') + 1])
        index = len(calls)
        calls.append(command)
        debug = directory / 'claude_image_tools' / 'smoke'
        debug.mkdir(parents=True)
        events = [{'event': 'attempt_started', 'attempt': 1}]
        if index > 0:
            events += [{'event': 'retry_scheduled', 'retry_decision': 'retry_scheduled',
                        'category': 'STREAM_EOF'}, {'event': 'attempt_started', 'attempt': 2}]
        if index == 1:
            events += [{'event': 'retry_recovered'}]
        if index == 2:
            events += [{'event': 'attempts_exhausted', 'retry_decision': 'attempts_exhausted',
                        'category': 'STREAM_EOF'}]
        (debug / 'responses_diagnostics.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        (debug / 'stdout.log').write_text(json.dumps({'type': 'result', 'subtype': 'success' if index < 2 else 'error_responses'}))
        return SimpleNamespace(returncode=0 if index < 2 else 1)

    monkeypatch.setattr(check.subprocess, 'run', run)
    output = tmp_path / 'report'
    assert check.main(['--local-responses', '--rounds', '3', '--output-dir', str(output)]) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['passed'] == 2 and report['failed'] == 1
    assert report['first_attempt_passed'] == report['passed_after_retry'] == 1
    assert report['rounds'][1]['request_retry_count'] == 1
    assert report['rounds'][1]['recovered_request_count'] == 1
    assert all('--local-responses' in command for command in calls)
