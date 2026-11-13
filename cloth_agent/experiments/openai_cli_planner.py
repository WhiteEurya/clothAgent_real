"""One offline image decision via Codex CLI. No camera or robot imports."""
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import jsonschema
from . import unfold_console as console


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(folder, body):
    """Preserve all supplied text and image order; never load project context."""
    if body.get('tools'):
        raise ValueError('CLI image planner expects a tool-free request')
    parts = ['Offline image planning only. Do not use tools, inspect files, execute commands, or control hardware. Return only the requested JSON.', body.get('instructions', '')]
    images = []
    for message in body['input']:
        if message.get('role', 'user') != 'user':
            raise ValueError('Unsupported planner message role')
        for item in message['content']:
            if item['type'] == 'input_text':
                parts.append(item['text'])
            elif item['type'] == 'input_image':
                url = item['image_url']
                prefix, data = url.split(',', 1)
                extensions = {'data:image/png;base64': '.png', 'data:image/jpeg;base64': '.jpg'}
                if prefix not in extensions:
                    raise ValueError('Planner requires embedded PNG/JPEG images')
                path = folder / f'image_{len(images)+1}{extensions[prefix]}'
                path.write_bytes(base64.b64decode(data, validate=True))
                images.append(path)
                parts.append(f'[Attached image {len(images)}]')
            else:
                raise ValueError('Unsupported planner input content')
    schema = body['text']['format']['schema']
    save(folder / 'schema.json', schema)
    prompt = '\n'.join(parts)
    (folder / 'cli_prompt.txt').write_text(prompt, encoding='utf-8')
    return prompt, images, schema


def validate_events(events):
    """A transient CLI reconnect may precede a successful terminal event."""
    completed=[i for i,e in enumerate(events) if e.get('type')=='turn.completed']
    tools=[e for e in events if e.get('type') in ('item.started','item.completed')
           and e.get('item',{}).get('type') not in ('agent_message','reasoning')]
    if tools or len(completed)!=1 or any(e.get('type')=='turn.failed' for e in events):
        raise ValueError('CLI planning did not finish one tool-free turn; see events.jsonl')
    recovered=[]
    for index,event in enumerate(events):
        if event.get('type')!='error':continue
        message=event.get('message','')
        if (index<completed[0] and isinstance(message,str)
                and message.startswith('Reconnecting... ')
                and 'stream disconnected before completion:' in message):
            recovered.append(message)
        else:
            raise ValueError('CLI planning contains an unresolved error; see events.jsonl')
    return events[completed[0]],recovered


def failed_turn_message(path):
    """Recover the service error that the CLI otherwise hides behind exit 1."""
    try:
        events=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception:
        return None
    for event in reversed(events):
        if event.get('type')=='turn.failed':
            error=event.get('error')
            if isinstance(error,dict) and error.get('message'):return str(error['message'])
            if error:return str(error)
        if event.get('type')=='error' and event.get('message'):return str(event['message'])
    return None


def direct_once(folder, body, headers, timeout_s):
    """Compatible with existing direct_once callback; one CLI turn, no host retry."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'attempt.json').open('x', encoding='utf-8') as f:
        json.dump({'started_at': datetime.now(timezone.utc).isoformat(), 'transport': 'openai_cli', 'cli_attempts': 1, 'retries': 0}, f)
    save(folder / 'request_payload.json', body)
    result = dict(status='FAILED', transport='openai_cli', cli_attempts=1, retries=0,
                  model_response_count=0, tool_calls=0, model=body['model'],
                  reasoning={'effort': body.get('reasoning', {}).get('effort', 'high')})
    started = time.perf_counter()
    process = None
    try:
        prompt, images, schema = prepare(folder, body)
        with tempfile.TemporaryDirectory(prefix='cloth_cli_planner_') as work:
            command = ['codex', 'exec', '-p', 'openai', '-m', body['model'],
                       '-c', 'model_reasoning_effort='+json.dumps(result['reasoning']['effort']),
                       '--ephemeral', '--skip-git-repo-check', '-s', 'read-only', '-C', work,
                       '--json', '--output-schema', str(folder/'schema.json'),
                       '-o', str(folder/'raw_output.txt')]
            for image in images:
                command.extend(['-i', str(image)])
            command.append('-')
            save(folder/'command.json', command)
            with (folder/'events.jsonl').open('w', encoding='utf-8') as out, (folder/'stderr.txt').open('w', encoding='utf-8') as err:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                           text=True, encoding='utf-8', start_new_session=True)
                try:
                    console.watch_model(folder/'events.jsonl')
                    with console.stage('模型正在看图并规划下一步','model'):
                        process.communicate(prompt, timeout=timeout_s)
                finally:
                    console.poll_model()
                    console.watch_model(None)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            result['exit_code'] = process.returncode
            if process.returncode:
                detail=failed_turn_message(folder/'events.jsonl')
                if detail:result['service_error']=detail
                raise RuntimeError(f'Codex CLI exited {process.returncode}'+
                                   (f': {detail}' if detail else '; see stderr.txt'))
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        completed = [e for e in events if e.get('type') == 'turn.completed']
        items = [e['item'] for e in events if e.get('type') == 'item.completed']
        tools = [e for e in events if e.get('type') in ('item.started', 'item.completed') and e.get('item', {}).get('type') not in ('agent_message', 'reasoning')]
        result['tool_calls'] = len(tools)
        result['model_response_count'] = len(completed)
        completed_event,recovered=validate_events(events)
        result['recovered_transport_errors']=recovered
        result['transport_reconnect_events']=len(recovered)
        value = json.loads((folder/'raw_output.txt').read_text(encoding='utf-8'))
        jsonschema.validate(value, schema)
        result.update(status='VALID_JSON', usage=completed_event.get('usage'), configured_model=body['model'])
        save(folder/'raw_response.json', {'transport': 'openai_cli', 'items': items, 'usage': result['usage'],
                                        'recovered_transport_errors':recovered})
        if recovered:console.emit('连接已恢复','CLI 重连后正常完成，动作 JSON 检查通过。')
        return value, result
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['wall_latency_s'] = time.perf_counter()-started
        save(folder/'model_metrics.json', result)
