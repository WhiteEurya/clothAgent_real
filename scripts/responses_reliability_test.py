#!/usr/bin/env python3
"""Repeat saved-RGB tool integration checks without robot/camera connections.

Uses remote_image_tools_test.py unchanged for every round. This measures the
image-tool/API path, not a complete physical folding task or semantic accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def summarize_round(directory):
    debug = directory / 'claude_image_tools' / 'smoke'
    diagnostics = debug / 'responses_diagnostics.jsonl'
    events = []
    if diagnostics.exists():
        for line in diagnostics.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    final = {}
    stdout = debug / 'stdout.log'
    if stdout.exists():
        with stdout.open() as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get('type') == 'result':
                    final = value
    return dict(
        http_attempt_count=sum(e.get('event') == 'attempt_started' for e in events),
        request_retry_count=sum(e.get('event') == 'attempt_started' and e.get('attempt', 1) > 1 for e in events),
        recovered_request_count=sum(e.get('event') == 'retry_recovered' for e in events),
        failures=[{k: e.get(k) for k in ('category', 'http_status', 'client_request_id',
                  'response_id', 'request_index', 'attempt', 'retry_decision', 'upstream_error')}
                  for e in events if e.get('retry_decision')],
        terminal_subtype=final.get('subtype'), model=final.get('model'),
        reasoning_effort=final.get('reasoning_effort'), response_count=final.get('response_count'),
        tool_call_count=final.get('num_tool_calls'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, help='saved PNG; omitted: deterministic synthetic color chart')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--timeout-s', type=int, default=600, help='per round; includes all API retries')
    parser.add_argument('--host', default='company-planner')
    parser.add_argument('--local-responses', action='store_true', help='bypass SSH and relay on the company host')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 50 or args.timeout_s < 1:
        parser.error('rounds must be 1..50 and timeout-s must be positive')
    image = args.image.resolve(strict=True) if args.image else None
    output = (args.output_dir or Path('results/responses_reliability') /
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'requested_rounds': args.rounds,
              'transport': 'local_company_shell' if args.local_responses else 'https_and_ssh',
              'scope': 'synthetic/saved RGB tools and JSON; no robot or complete folding validation',
              'rounds': []}
    report_path = output / 'report.json'

    def save():
        rounds = report['rounds']
        report.update(completed_rounds=len(rounds), passed=sum(r['passed'] for r in rounds),
                      first_attempt_passed=sum(r['passed'] and not r['request_retry_count'] for r in rounds),
                      passed_after_retry=sum(r['passed'] and r['request_retry_count'] > 0 for r in rounds),
                      failed=sum(not r['passed'] for r in rounds))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    save()
    print(f'Report: {report_path}', flush=True)
    try:
        for index in range(1, args.rounds + 1):
            directory = output / f'round_{index:03d}'
            command = [sys.executable, str(Path(__file__).with_name('remote_image_tools_test.py')),
                       '--timeout-s', str(args.timeout_s), '--output-dir', str(directory)]
            command += ['--local-responses'] if args.local_responses else ['--host', args.host]
            if image:
                command.append(str(image))
            started = time.monotonic()
            print(f'Round {index}/{args.rounds}: started', flush=True)
            with (output / f'round_{index:03d}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            row = {'round': index, 'passed': result.returncode == 0, 'returncode': result.returncode,
                   'elapsed_s': round(time.monotonic() - started, 3), **summarize_round(directory)}
            report['rounds'].append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report['status'] = 'PASSED' if not report['failed'] else 'FAILED'
    except KeyboardInterrupt:
        report['status'] = 'INTERRUPTED'
        raise
    finally:
        save()
    print(f"{report['status']}: {report['passed']}/{args.rounds}; report: {report_path}", flush=True)
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
