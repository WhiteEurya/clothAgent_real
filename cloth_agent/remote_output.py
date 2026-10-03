"""Standalone remote CLI wrapper: spool, validate, then send compact JSONL."""
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
    event_times = []
    with raw.open('wb') as output:
        if not record_timing:
            result = subprocess.run(argv, stdout=output, **({"input": model_input} if model_input is not None else {}))
        else:
            # Timestamp on the producer host before the validated spool is sent
            # over SSH. These are CLI emission times, not internal model times.
            started = time.monotonic()
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
            for line in process.stdout:
                if line.strip():
                    event_times.append(time.monotonic() - started)
                output.write(line)
            process.stdout.close()
            process.wait()
            if writer is not None:
                writer.join()
            result = subprocess.CompletedProcess(argv, process.returncode)
            print('__CLOTH_PROFILE__ ' + json.dumps({
                'process_elapsed_s': time.monotonic() - started,
                'terminal_event_elapsed_s': event_times[-1] if event_times else None,
                'event_count': len(event_times),
                'clock': 'remote_monotonic_cli_line_receipt'}), file=sys.stderr, flush=True)
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
    for index, event in enumerate(events):
        # Terminal payload is the authoritative result and must stay unchanged.
        payload = event if event.get('type') == 'result' else compact(event)
        if record_timing and event.get('type') != 'result':
            payload = {**payload, '_cloth_timing': {
                'elapsed_s': event_times[index], 'clock': 'remote_monotonic_cli_line_receipt'}}
        print(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), flush=True)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
