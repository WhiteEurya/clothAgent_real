"""Build an offline, evidence-linked reader of the archived physical run."""
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
import math
from PIL import Image

OUT = Path(__file__).resolve().parent
RUN = Path('/mnt/newssd/sja/clothAgent_real/runs/2026-09-24/fold_20260924T014204289928777Z')

def read(p, fallback=None):
    return json.loads(p.read_text()) if p.exists() else fallback

def lines(p):
    if not p.exists():
        return []
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def raw(p):
    return p.read_text() if p.exists() else '此项未保存'

def photo(p):
    if not p.exists():
        return None
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:24]
    target = OUT / 'images' / (digest + '.jpg')
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with Image.open(p) as im:
            im.thumbnail((420, 420))
            im.convert('RGB').save(target, quality=80)
    return {'src': 'images/' + target.name, 'source': str(p), 'name': p.name}

def compact_images(value):
    if isinstance(value, dict):
        if value.get('type') == 'base64' and 'data' in value:
            return {**value, 'data': '[图片二进制省略；请查看对应调用的原始图片/returned_images]'}
        return {k: compact_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact_images(v) for v in value]
    return value

def browser_json(value):
    """Keep non-finite measurements explicit while emitting strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value).replace('nan', 'NaN').replace('inf', 'Infinity')
    if isinstance(value, dict):
        return {k: browser_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [browser_json(v) for v in value]
    return value

rows = []
for segment in sorted((RUN / 'results/fold_exploration').iterdir()):
    events = lines(segment / 'debug_events.jsonl')
    by_iteration = {}
    current = 0
    for ev in events:
        if ev['stage'] == 'iteration' and ev['message'].startswith('starting iteration'):
            current = ev['fields']['iteration']
        by_iteration.setdefault(current, []).append(ev)
    for directory in sorted(segment.glob('iteration_*')):
        n = int(directory.name.split('_')[-1])
        record = read(directory / 'record.json', read(directory / 'partial_record.json', {}))
        row = {'number': len(rows) + 1, 'segment': segment.name, 'iteration': n,
               'status': record.get('status'), 'step': record.get('planned_step'),
               'source': str(directory), 'calls': [], 'events': [], 'files': {}}
        for reqpath in directory.glob('claude_image_tools/*/request.json'):
            d = reqpath.parent
            req = read(reqpath)
            stream = lines(d / 'claude_events.jsonl')
            stamped = [e for e in stream if e.get('received_at')]
            start = ''
            if stamped:
                e = stamped[0]
                start = (datetime.fromisoformat(e['received_at']) - timedelta(seconds=e.get('received_elapsed_s', 0))).isoformat()
            result = read(d / 'claude_result.json', {})
            messages = []
            for envelope in stream:
                ev = envelope.get('event', {})
                if ev.get('type') in ('assistant', 'user'):
                    content = ev.get('message', {}).get('content', [])
                    # Keep observable conversation and tools, omit private thinking blocks.
                    content = [c for c in content if c.get('type') not in ('thinking', 'redacted_thinking')]
                    if content:
                        messages.append({'role': ev['type'], 'content': compact_images(content)})
            call = {'name': d.name, 'time': start, 'source': str(d),
                    'prompt': raw(d / 'prompt.txt'), 'request_prompt': req.get('prompt'),
                    'system': req.get('system_prompt'), 'schema': req.get('schema'),
                    'images_manifest': req.get('image_paths'),
                    'result': result, 'messages': messages,
                    'tools': lines(d / 'images/image_tool_calls.jsonl'),
                    'timing': read(d / 'timing.json', {}), 'images': []}
            for p in sorted((d / 'images').glob('image_*.*')):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    call['images'].append(photo(p))
            row['calls'].append(call)
        row['calls'].sort(key=lambda c: c['time'] or '9999')
        row['events'] = [e for e in by_iteration.get(n, []) if not e['stage'].startswith('remote-')]
        for name in ['record.json', 'planning_diagnostics.json', 'supervisor_before.json',
                     'claude_plan.json', 'host_compilation.json', 'execution_plan.json',
                     'execution.json', 'evaluation.json', 'claude_evaluation_result.json',
                     'supervisor_after.json', 'failure_detection.json', 'height_retry_followup.json',
                     'experience_update/request.json', 'experience_update/response.json',
                     'experience_update/store_receipt.json']:
            if (directory / name).exists():
                row['files'][name] = ({'source': str(directory / name), 'note': '完整 record 含递归历史，使用原始证据链接查看；本轮产物分别列于下方。'}
                                      if name == 'record.json' else read(directory / name))
        row['grounding'] = {str(p.relative_to(directory)): read(p) for p in directory.glob('planning_attempt*/*/*.json')}
        rows.append(row)

data = {'run': str(RUN), 'rows': rows,
        'segments': [{k: v for k, v in read(p).items() if k in ('status', 'created_at', 'output_dir')}
                     for p in sorted((RUN/'results/fold_exploration').glob('*/summary.json'))],
        'restarts': lines(RUN/'unattended_restarts.jsonl')}
data = browser_json(data)
(OUT / 'flow_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
template = (OUT / 'template.html').read_text()
(OUT / 'index.html').write_text(template.replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')))
print(json.dumps({'iterations': len(rows), 'calls': sum(len(r['calls']) for r in rows),
                  'missing_call_time': sum(not c['time'] for r in rows for c in r['calls']),
                  'html_bytes': (OUT/'index.html').stat().st_size}, indent=2))
