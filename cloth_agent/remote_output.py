"""Standalone remote CLI wrapper: spool, validate, then send compact JSONL."""
import json
import os
from pathlib import Path
import subprocess
import sys


def compact(value):
    """Remove only an exact duplicate of message tool-result content.

    The message copy is required by local image delivery/identity validation.
    Never discard unique image evidence or change terminal results.
    """
    if not isinstance(value, dict) or value.get('type') == 'result':
        return value
    message = value.get('message')
    if not isinstance(message, dict) or 'tool_use_result' not in value:
        return value
    content = message.get('content')
    if not isinstance(content, list):
        return value
    duplicate = value['tool_use_result']
    if any(isinstance(block, dict) and block.get('type') == 'tool_result'
           and block.get('content') == duplicate for block in content):
        return {key: item for key, item in value.items() if key != 'tool_use_result'}
    return value


def main(argv):
    model_input = None
    if argv and argv[0] == '--context-envelope':
        argv = argv[1:]
        envelope = json.load(sys.stdin)
        directory = Path('context')
        directory.mkdir(exist_ok=True)
        for name, content in envelope['files'].items():
            if Path(name).name != name or not name.endswith('.json'):
                raise ValueError('invalid context file name')
            (directory / name).write_text(content, encoding='utf-8')
        model_input = envelope['prompt'].encode('utf-8')
    raw = Path('claude_raw.jsonl')
    with raw.open('wb') as output:
        result = subprocess.run(argv, stdout=output, **({"input": model_input} if model_input is not None else {}))
    # stdout now belongs only to this synchronous writer, not the model runtime.
    os.set_blocking(sys.stdout.fileno(), True)
    os.set_blocking(sys.stderr.fileno(), True)
    try:
        events = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        if not events or any(not isinstance(event, dict) for event in events) or events[-1].get('type') != 'result':
            raise ValueError('missing terminal result')
    except (ValueError, UnicodeError):
        saved = raw.resolve().parent.with_suffix('.failed.jsonl')
        raw.replace(saved)
        print(f'REMOTE_OUTPUT_INVALID: exit={result.returncode}; raw={saved}', file=sys.stderr, flush=True)
        return result.returncode or 65
    if result.returncode:
        saved = raw.resolve().parent.with_suffix('.failed.jsonl')
        raw.replace(saved)
        print(f'REMOTE_CLI_FAILED: exit={result.returncode}; raw={saved}', file=sys.stderr, flush=True)
    print(f'REMOTE_OUTPUT_VALID: events={len(events)}', file=sys.stderr, flush=True)
    for event in events:
        # Terminal payload is the authoritative result and must stay unchanged.
        payload = event if event.get('type') == 'result' else compact(event)
        print(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), flush=True)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
