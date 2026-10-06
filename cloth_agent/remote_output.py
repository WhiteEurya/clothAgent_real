"""Remote CLI wrapper: stream diagnostic events, spool raw output, validate terminal results."""
import json
import base64
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


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


def multimodal_message(prompt, images):
    """Attach actual image bytes to CLI stream input, with no model file reads."""
    content = [{"type": "text", "text": prompt}]
    for index, path in enumerate(images):
        content.append({"type": "text", "text": f"Attached image_{index}"})
        content.append({"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": base64.b64encode(Path(path).read_bytes()).decode()}})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def main(argv):
    record_timing = bool(argv and argv[0] == '--record-event-timing')
    if record_timing:
        argv = argv[1:]
    model_input = None
    if argv and argv[0] == '--text-only':
        argv = argv[1:]
        model_input = multimodal_message(sys.stdin.read(), []).encode()
    if argv and argv[0] == '--direct-images':
        count = int(argv[1])
        if not 1 <= count <= 64:
            raise ValueError('invalid direct image count')
        argv = argv[2:]
        model_input = multimodal_message(sys.stdin.read(), [Path(f'image_{i}.png') for i in range(count)]).encode()
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
    started = time.monotonic()
    state = {'event_count': 0, 'last_event_elapsed_s': None, 'last_event_type': None}
    stopped = threading.Event()

    def emit(payload, *, diagnostic=False):
        stream = sys.stderr if diagnostic else sys.stdout
        os.set_blocking(stream.fileno(), True)
        prefix = '__CLOTH_PROGRESS__ ' if diagnostic else ''
        print(prefix + json.dumps(payload, ensure_ascii=False, separators=(',', ':')), file=stream, flush=True)

    def heartbeat():
        while not stopped.wait(5):
            elapsed = time.monotonic() - started
            last = state['last_event_elapsed_s']
            emit({'phase': 'waiting_for_cli', 'elapsed_s': elapsed, **state,
                  'idle_s': elapsed - last if last is not None else elapsed}, diagnostic=True)

    terminal = None
    invalid = False
    last_type = None
    emit({'phase': 'cli_started', 'elapsed_s': 0., **state}, diagnostic=True)
    process = subprocess.Popen(argv, stdout=subprocess.PIPE,
                               **({'stdin': subprocess.PIPE} if model_input is not None else {}))
    def write_input():
        try:
            process.stdin.write(model_input)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    writer = None
    if model_input is not None:
        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()
    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    try:
        with raw.open('wb') as output:
            for line in process.stdout:
                output.write(line)
                output.flush()
                if not line.strip():
                    continue
                elapsed = time.monotonic() - started
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError('CLI event must be an object')
                except (ValueError, UnicodeError):
                    invalid = True
                    last_type = 'invalid'
                    # Keep malformed bytes as diagnostics, never as a model result.
                    emit({'phase': 'invalid_cli_line', 'elapsed_s': elapsed,
                          'raw_base64': base64.b64encode(line).decode()}, diagnostic=True)
                    continue
                nested = event.get('event') if isinstance(event.get('event'), dict) else {}
                delta = nested.get('delta') if isinstance(nested.get('delta'), dict) else {}
                label = delta.get('type') or nested.get('type') or event.get('subtype') or event.get('type')
                state.update(event_count=state['event_count']+1, last_event_elapsed_s=elapsed,
                             last_event_type=label)
                last_type = event.get('type')
                if last_type == 'result':
                    terminal = event
                    continue
                payload = compact(event)
                if record_timing:
                    payload = {**payload, '_cloth_timing': {
                        'elapsed_s': elapsed, 'clock': 'remote_monotonic_cli_line_receipt'}}
                emit(payload)
        process.stdout.close()
        process.wait()
        if writer is not None:
            writer.join(timeout=1)
    finally:
        stopped.set()
        pulse.join(timeout=1)
    emit({'phase': 'cli_finished', 'elapsed_s': time.monotonic()-started,
          'returncode': process.returncode, 'terminal_received': terminal is not None, **state}, diagnostic=True)
    if record_timing:
        print('__CLOTH_PROFILE__ ' + json.dumps({
            'process_elapsed_s': time.monotonic()-started,
            'terminal_event_elapsed_s': state['last_event_elapsed_s'],
            'event_count': state['event_count'],
            'clock': 'remote_monotonic_cli_line_receipt'}), file=sys.stderr, flush=True)
    if invalid or last_type != 'result' or terminal is None or process.returncode:
        saved = raw.resolve().parent.with_suffix('.failed.jsonl')
        raw.replace(saved)
        label = 'REMOTE_CLI_FAILED' if process.returncode and terminal is not None else 'REMOTE_OUTPUT_INVALID'
        print(f'{label}: exit={process.returncode}; raw={saved}', file=sys.stderr, flush=True)
        # A terminal-looking object from a failing process is diagnostic only.
        if terminal is not None:
            emit({'phase': 'unaccepted_terminal', 'result': terminal}, diagnostic=True)
        return process.returncode or 65
    print(f'REMOTE_OUTPUT_VALID: events={state["event_count"]}', file=sys.stderr, flush=True)
    emit(terminal)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
