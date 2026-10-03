"""Two visual decisions: bind an existing skill once, execute locally, then verify."""
from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

from ..image_tools_mcp import ImageTools
from .common import canonical, digest, read_json, write_json
from .information_probe import ROOT, TASKS, RESULT_SCHEMA, compact_skill, prepare_observation, validate_result
from .model import RuntimeClaude
from .policy import PolicyError, obj, validate_schema

BINDING_SCHEMA = obj({
    'roi': {'anyOf': [{'type': 'null'}, {'type': 'array', 'minItems': 4, 'maxItems': 4,
           'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}]},
    'paired_crop': {'type': 'boolean'}, 'enlarge': {'type': 'boolean'},
    'target_description': {'type': 'string', 'minLength': 1, 'maxLength': 300},
    'reason': {'type': 'string', 'minLength': 1, 'maxLength': 300},
})
STEPS = [
    {'id': 'clean_crop', 'operation': 'crop_image', 'source': 'clean', 'condition': 'has_roi'},
    {'id': 'overlay_crop', 'operation': 'crop_image', 'source': 'overlay', 'condition': 'paired_crop'},
    {'id': 'clean_large', 'operation': 'resize_image', 'source': 'clean_crop', 'condition': 'enlarge'},
    {'id': 'overlay_large', 'operation': 'resize_image', 'source': 'overlay_crop', 'condition': 'paired_enlarge'},
]


def execute_recipe(obs, images, binding, recipe, output):
    """Only finite image edits from this supported recipe. No model calls here."""
    validate_schema(binding, BINDING_SCHEMA)
    if recipe['steps'] != STEPS:
        raise PolicyError('Unsupported recipe; host supports the declared crop/resize flow only')
    side, maximum = recipe['display_long_side'], recipe['max_scale']
    if type(side) is not int or not 64 <= side <= 2048 or type(maximum) not in (int, float) or not 1 <= maximum <= 4:
        raise PolicyError('Invalid display scale budget')
    roi = binding['roi']
    if roi is not None and not (roi[0] < roi[2] and roi[1] < roi[3]):
        raise PolicyError('Empty or reversed ROI')
    if roi is None and (binding['paired_crop'] or binding['enlarge']):
        raise PolicyError('Crop-dependent conditions require ROI')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for i, path in enumerate(images):
        shutil.copyfile(path, output / f'image_{i}.png')
    tools = ImageTools(output, len(images), edit_limit=4)
    roots = {role: [i for i in obs['images'] if i['role'] == role] for role in ('clean', 'overlay')}
    if any(len(v) != 1 for v in roots.values()):
        raise PolicyError('Ambiguous source pair')
    if roots['clean'][0]['size'] != roots['overlay'][0]['size']:
        raise PolicyError('Pair size mismatch')
    bindings = {role: tools.views[items[0]['image_id']] for role, items in roots.items()}
    conditions = {'has_roi': roi is not None, 'paired_crop': binding['paired_crop'],
                  'enlarge': binding['enlarge'], 'paired_enlarge': binding['paired_crop'] and binding['enlarge']}
    log = []
    for step in recipe['steps']:
        if not conditions[step['condition']]:
            log.append({**step, 'status': 'SKIPPED_CONDITION'}); continue
        source = bindings[step['source']]
        if step['operation'] == 'crop_image':
            w, h = source['size']
            args = {'box': [math.floor(roi[0]*w), math.floor(roi[1]*h), math.ceil(roi[2]*w), math.ceil(roi[3]*h)]}
        else:
            scale = min(maximum, side / max(source['size']))
            if scale <= 1:
                log.append({**step, 'status': 'SKIPPED_ALREADY_LARGE'}); continue
            args = {'scale': scale}
        result = tools.call(step['operation'], {'image_id': source['image_id'], **args})
        if not isinstance(result, dict) or 'image_id' not in result:
            raise PolicyError('Image operation did not produce a view')
        bindings[step['id']] = result
        log.append({**step, 'status': 'EXECUTED', 'arguments': args,
                    'view': {k: result[k] for k in ('image_id', 'path', 'size', 'to_original')}})
    catalog = [{**i, 'source_view_id': i['image_id']} for i in obs['images']]
    paths = [output / f'image_{i}.png' for i in range(len(images))]
    for entry in log:
        if entry['status'] != 'EXECUTED': continue
        view = entry['view']
        catalog.append({'image_id': f'image_{len(paths)}', 'role': entry['id'], 'size': view['size'],
                        'source_view_id': view['image_id'], 'to_original': view['to_original']})
        paths.append(Path(view['path']))
    write_json(output / 'execution.json', {'binding': binding, 'recipe': recipe, 'steps': log, 'catalog': catalog})
    return paths, catalog, log


def prepare_skill_views(model, obs, images, artifact, output, *, timeout=360, information_tasks=None):
    """Reusable preprocessing hook: one binding call, local execution, no verification call."""
    output = Path(output); start = time.monotonic()
    skill, recipe = compact_skill(artifact), artifact['execution_recipe']
    data = {'observation': obs, 'information_tasks': TASKS if information_tasks is None else information_tasks,
            'skill': skill, 'execution_recipe': recipe}
    prompt = ('Use the attached CURRENT RGB to bind this already-defined skill, not design a new workflow. '
        'Return only ROI and branch conditions. ROI is normalized [left,top,right,bottom] in the catalog image with role=clean; '
        'select a visible boundary relevant to the supplied current task, retaining context. Do not automatically '
        'prefer the label, left end or seam if the task concerns another structure. If originals already suffice or '
        'crop cannot help, roi=null and both flags=false. paired_crop only if local clean/overlay comparison is needed; '
        'enlarge only for display legibility. No historical coordinates, tool calls or prose plan.\n' + canonical(data))
    print('[skill] bind current ROI/conditions once', flush=True)
    binding = model.invoke(prompt=prompt, schema=BINDING_SCHEMA, images=images,
        output=output / 'bind', stage='information_skill_bind', timeout_s=timeout)
    write_json(output / 'binding.json', binding)
    if time.monotonic()-start >= timeout:
        raise TimeoutError('Preprocessing budget exhausted before execution')
    print('[skill] execute recipe locally; no model calls between operations', flush=True)
    local_start = time.monotonic()
    paths, catalog, log = execute_recipe(obs, images, binding, recipe, output / 'execution')
    metrics = {'host_execution_s': time.monotonic()-local_start,
               'host_image_ops': sum(s['status']=='EXECUTED' for s in log),
               'preprocessing_s': time.monotonic()-start}
    write_json(output / 'preprocessing.json', metrics)
    return paths, catalog, log, metrics


def run_direct(model, obs, images, artifact, output, timeout=360):
    output = Path(output); start = time.monotonic()
    skill = compact_skill(artifact); recipe = artifact['execution_recipe']
    report = {'status': 'RUNNING', 'mode': 'bind_execute_verify', 'semantic_stage_budget': 2,
              'replanning_between_operations': False, 'skill_hash': digest(artifact), 'observation_hash': digest(obs)}
    write_json(output / 'report.json', report)
    try:
        paths, catalog, log, metrics = prepare_skill_views(model, obs, images, artifact, output,
                                                          timeout=timeout)
        report.update(metrics)
        prompt = ('Inspect the actual attached originals and host-produced views. Complete every requested '
            'information task using only visible evidence. This is final verification; no tools or replanning. '
            'Positive and negative judgments can both satisfy a task. If evidence remains insufficient return UNKNOWN '
            'with missing_information; do not infer absence or continuity through occlusion. No robot plan. '
            'Only claim operations in the supplied executed_steps as performed. Source IDs use the attached catalog. '
            'Return concise Chinese JSON.\n' + canonical({'information_tasks': TASKS, 'images': catalog,
            'skill': skill, 'executed_steps': [{k:v for k,v in s.items() if k!='view'} for s in log]}))
        print('[direct] verify all information once', flush=True)
        result = model.invoke(prompt=prompt, schema=RESULT_SCHEMA, images=paths,
            output=output / 'verify', stage='information_skill_verify', timeout_s=max(0, timeout-(time.monotonic()-start)))
        write_json(output / 'result.json', result); validate_result(result)
        if any(i not in {c['image_id'] for c in catalog} for f in result['findings'] for i in f['source_image_ids']):
            raise PolicyError('Result cites an unknown attached image')
        report.update(status='COMPLETED', satisfied=sum(f['status']=='SATISFIED' for f in result['findings']),
                      unknown=sum(f['status']=='UNKNOWN' for f in result['findings']),
                      quality='MODEL_REPORTED_NOT_INDEPENDENTLY_VERIFIED')
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        report.update(elapsed_s=time.monotonic()-start, calls=model.calls,
                      model_invocations=len(model.calls),
                      exploratory_tool_round_trips=sum(c.get('exploratory_tool_round_trips',0) or 0 for c in model.calls))
        write_json(output / 'report.json', report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'results/harness_image_processing_20261002/manifest.json')
    p.add_argument('--decision-id')
    p.add_argument('--skill', type=Path, default=ROOT/'data/skills/experimental/visual_information.json')
    p.add_argument('--ssh-host', default='company-planner'); p.add_argument('--model', default='claude-opus-5')
    p.add_argument('--timeout', type=int, default=360)
    a=p.parse_args(argv)
    if a.timeout<=0:p.error('Positive timeout required')
    a.output.mkdir(parents=True,exist_ok=False)
    artifact=read_json(a.skill)
    obs,images,source=prepare_observation(read_json(a.manifest),a.output,a.decision_id)
    write_json(a.output/'skill_snapshot.json',artifact);write_json(a.output/'source.json',source)
    model=RuntimeClaude(backend='remote',ssh_host=a.ssh_host,model=a.model,timeout_s=a.timeout,max_turns=4)
    result=run_direct(model,obs,images,artifact,a.output,a.timeout)
    print(canonical({k:v for k,v in result.items() if k!='calls'}),flush=True)
    return 0 if result['status']=='COMPLETED' else 1


if __name__=='__main__':
    raise SystemExit(main())
