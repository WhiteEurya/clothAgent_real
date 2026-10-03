"""Independent, read-only visual information A/B pilot; no compiler or robot."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from ..image_tools_mcp import pixel_hash
from ..planner_backend import RemoteClaudeBackend, claude_result_envelope, parse_claude_json
from .common import digest, read_json, write_json
from .model import stream_tool_calls
from .policy import PolicyError, obj, validate_schema

ROOT = Path(__file__).resolve().parents[2]
TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 800}
TASKS = [
    {'id': 'global_layout', 'information_need': '描述当前衣物可见的整体轮廓、朝向、卷曲和堆叠区域。'},
    {'id': 'local_boundary', 'information_need': '在当前衣物一处可见的疑似缝线或折边处，判断缝线是否存在、是否连续及其与布料边缘的位置关系。明确指出所检查的区域。'},
    {'id': 'overlay_occlusion', 'information_need': '判断目标局部是否被 overlay 标记遮挡，并区分真实结构与叠加标记。'},
    {'id': 'reference_shape', 'information_need': '比较当前衣物与参考目标的可见外轮廓、宽高趋势和紧凑度；内部结构不可辨时明确说明。'},
]
RESULT_SCHEMA = obj({'findings': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': obj({
    'need_id': {'enum': [t['id'] for t in TASKS]},
    'status': {'enum': ['SATISFIED', 'UNKNOWN']},
    'answer': TEXT,
    'source_image_ids': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'uniqueItems': True, 'items': TEXT},
    'method_used': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': TEXT},
    'success_check': TEXT,
    'missing_information': {'type': 'string', 'maxLength': 800},
})}})
SYSTEM = '''You inspect saved garment RGB images to obtain requested visual information. No robot action,
no grasp selection, no motion plan, no external lookup. Only inspect supplied images with the image tools.
Use actual returned pixels, not filenames or metadata, for visual assertions. Treat annotations as hints,
not ground truth. SATISFIED means enough visibility to judge, including a justified negative finding.
UNKNOWN means insufficient evidence, never absence. State the missing information. Do not claim an
unexecuted crop/view as performed. The supplied experimental skill is guidance, not scene evidence.
Return concise Chinese JSON covering every requested need exactly once. Do not read any unrelated files.'''


class ProbeBackend(RemoteClaudeBackend):
    def _image_tool_setup(self, job, count):
        setup, flags, prompt = super()._image_tool_setup(job, count)
        # Keep user authentication/provider settings, as RuntimeClaude does; exclude project settings.
        return setup, flags + "--setting-sources user --no-chrome ", prompt


def compact_skill(artifact):
    if artifact.get('scope') != 'visual_information_acquisition':
        raise ValueError('Not a visual information skill')
    fields = ('id', 'information_need', 'applicable_when', 'method', 'success_check',
              'on_insufficient', 'limitations')
    skills = [{k: s[k] for k in fields} for s in artifact['skills']]
    if len({s['id'] for s in skills}) != len(skills):
        raise ValueError('Duplicate skill IDs')
    return {'status': artifact['status'], 'skills': skills}


def prepare_observation(manifest, output, decision_id=None):
    decisions = manifest['decisions']
    matches = [d for d in decisions if d['decision_id'] == decision_id] if decision_id else decisions[-1:]
    if len(matches) != 1:
        raise ValueError('Select exactly one decision from the manifest')
    trace = matches[0]
    before = trace['pre_decision']
    directory = Path(output) / 'obs'
    directory.mkdir(parents=True, exist_ok=False)
    images, catalog = [], []
    for image in before['images']:
        # Semantic hints are excluded equally from both arms; only current roots and references.
        if image.get('role') not in {'clean', 'overlay', 'reference'} or image.get('status') != 'AVAILABLE':
            continue
        with Image.open(image['path']) as im:
            if pixel_hash(im) != image['rgb_sha256'] or list(im.size) != image['size']:
                raise ValueError('Observation image changed')
            path = directory / f'image_{len(images)}.png'
            im.convert('RGB').save(path)
        name = Path(image.get('source', '')).name
        catalog.append({'image_id': path.stem, 'role': image['role'], 'size': image['size'],
                        'rgb_sha256': image['rgb_sha256'],
                        'reference_kind': ('target' if 'target' in name else 'source' if 'source' in name else 'unspecified')
                            if image['role'] == 'reference' else None})
        images.append(path.resolve())
    if not {'clean', 'overlay', 'reference'} <= {i['role'] for i in catalog}:
        raise ValueError('This pilot requires clean, overlay and reference roots')
    obs = {'observation_id': before['observation_id'], 'images': catalog}
    write_json(directory / 'observation.json', obs)
    return obs, images, {'decision_id': trace['decision_id'], 'iteration_id': trace['iteration_id']}


def validate_result(result):
    validate_schema(result, RESULT_SCHEMA)
    if {f['need_id'] for f in result['findings']} != {t['id'] for t in TASKS}:
        raise PolicyError('Findings must cover every information need exactly once')
    for f in result['findings']:
        if f['status'] == 'UNKNOWN' and not f['missing_information'].strip():
            raise PolicyError('UNKNOWN requires missing_information')


def tool_metrics(stdout, trace_directory):
    # RemoteClaudeBackend returns a compact result envelope. Full tool-use events
    # are saved separately by ImageDebugSession; count each unique tool-use ID.
    trace = Path(trace_directory) / 'claude_stdout.txt'
    names = stream_tool_calls(trace.read_text() if trace.is_file() else stdout)
    return {'tool_calls': dict(Counter(names)), 'tool_call_count': len(names),
            'image_edit_calls': sum(n.endswith(('__crop_image', '__resize_image', '__rotate_image')) for n in names),
            'tool_metrics_source': str(trace) if trace.is_file() else 'response.stdout'}


def run_arm(backend, obs, images, skill, output, *, model, timeout, edits, turns):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = {'observation': obs, 'information_tasks': TASKS}
    if skill is not None:
        payload['visual_information_skill'] = skill
    prompt = 'Complete these visual information tasks using the supplied current RGB images.\n' + json.dumps(payload, ensure_ascii=False)
    write_json(output / 'input.json', payload)
    report = {'status': 'RUNNING', 'prompt_chars': len(prompt), 'model_requested': model,
              'skill_present': skill is not None, 'image_count': len(images),
              'budget': {'seconds': timeout, 'image_edits': edits, 'model_turns': turns},
              'quality': 'MODEL_REPORTED_NOT_INDEPENDENTLY_VERIFIED'}
    write_json(output / 'report.json', report)
    start = time.monotonic()
    try:
        response = backend.invoke(prompt=prompt, image_paths=images, schema=RESULT_SCHEMA,
            system_prompt=SYSTEM, model=model, debug_dir=output / 'trace',
            image_edit_limit=edits, max_turns=turns, overall_timeout_s=timeout,
            usage_run_dir=output, usage_stage='visual_information_probe')
        (output / 'stdout.jsonl').write_text(response.stdout)
        (output / 'stderr.txt').write_text(response.stderr)
        envelope = claude_result_envelope(response.stdout)
        result = parse_claude_json(response.stdout)
        write_json(output / 'result.json', result)
        report.update(timings=response.timings, models=envelope.get('modelUsage'),
                      **tool_metrics(response.stdout, output / 'trace'))
        validate_result(result)
        report.update(status='COMPLETED', satisfied=sum(f['status'] == 'SATISFIED' for f in result['findings']),
                      unknown=sum(f['status'] == 'UNKNOWN' for f in result['findings']))
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                      timings=getattr(exc, 'timings', getattr(backend, 'last_timings', {})))
        for attr in ('stdout', 'stderr'):
            data = getattr(exc, attr, None)
            if data:
                (output / f'{attr}.txt').write_text(data.decode(errors='replace') if isinstance(data, bytes) else data)
    finally:
        report['elapsed_s'] = time.monotonic() - start
        write_json(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/harness_image_processing_20261002/manifest.json')
    parser.add_argument('--skill', type=Path, default=ROOT / 'data/skills/experimental/visual_information.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decision-id')
    parser.add_argument('--ssh-host', default='company-planner')
    parser.add_argument('--model', default='claude-opus-5')
    parser.add_argument('--timeout', type=int, default=360)
    parser.add_argument('--max-edits', type=int, choices=range(0, 25), default=4)
    parser.add_argument('--max-turns', type=int, choices=range(2, 21), default=12)
    parser.add_argument('--order', choices=['skill-first', 'baseline-first'], default='skill-first')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    artifact = read_json(args.skill)
    skill = compact_skill(artifact)
    args.output.mkdir(parents=True, exist_ok=False)
    obs, images, provenance = prepare_observation(read_json(args.manifest), args.output, args.decision_id)
    write_json(args.output / 'skill_snapshot.json', artifact)
    order = ['with_skill', 'without_skill'] if args.order == 'skill-first' else ['without_skill', 'with_skill']
    report = {'status': 'PREPARED', 'observation_hash': digest(obs), 'skill_hash': digest(artifact),
              'source': provenance, 'order': order, 'arms': {},
              'limitations': ['Single paired pilot; no statistical speedup claim.',
                  'Source observation may overlap skill provenance; not a held-out generalization test.',
                  'Network and provider/prompt cache may favor the second arm; inspect phase timings.',
                  'Information quality is model-reported and requires visual review.',
                  'This tests observation, not compiler latency or robot success.']}
    write_json(args.output / 'report.json', report)
    if args.prepare_only:
        print(json.dumps(report, ensure_ascii=False)); return 0
    for arm in order:
        print(f'[information probe] {arm}', flush=True)
        backend = ProbeBackend(ssh_host=args.ssh_host, timeout_s=args.timeout, image_tools=True)
        backend.progress_callback = lambda stage, event, duration_s=None, **kw: print(
            f'[probe] {stage}: {event}' + (f' ({duration_s:.1f}s)' if duration_s is not None else ''), flush=True)
        report['arms'][arm] = run_arm(backend, obs, images, skill if arm == 'with_skill' else None,
            args.output / arm, model=args.model, timeout=args.timeout, edits=args.max_edits, turns=args.max_turns)
        report['status'] = 'RUNNING'
        write_json(args.output / 'report.json', report)
    arms = report['arms']
    report['status'] = 'COMPLETED' if all(a['status'] == 'COMPLETED' for a in arms.values()) else 'FAILED'
    if report['status'] == 'COMPLETED':
        report['observed_wall_time_ratio_baseline_over_skill'] = arms['without_skill']['elapsed_s'] / arms['with_skill']['elapsed_s']
        report['speedup_established'] = False
    write_json(args.output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
