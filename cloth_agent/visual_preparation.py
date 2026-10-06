"""Evidence handoff between editable observation and read-only reasoning."""
from __future__ import annotations

import copy
import json
import hashlib
import shutil
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image

from .image_tools_mcp import pixel_hash, VERIFIED_DELIVERIES


def snapshot_preparation(debug_dir, images, context, payload):
    """Freeze the completed editing call before any reasoning call can run."""
    debug_dir = Path(debug_dir)
    destination = debug_dir.parent / (debug_dir.name + '_snapshot')
    staging = destination.with_name(destination.name + '.partial')
    shutil.copytree(debug_dir, staging)
    inputs = staging / 'original_inputs'
    inputs.mkdir()
    origins = []
    for index, path in enumerate(images):
        path = Path(path)
        target = inputs / f'image_{index}{path.suffix}'
        shutil.copy2(path, target)
        origins.append({'source': str(path), 'copy': str(target.relative_to(staging))})
    (staging / 'preparation_context.json').write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding='utf-8')
    (staging / 'preparation_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    files = []
    for path in sorted(staging.rglob('*')):
        if path.is_file():
            files.append({'path': str(path.relative_to(staging)), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (staging / 'snapshot_manifest.json').write_text(json.dumps({
        'source_directory': str(debug_dir), 'boundary': 'image_preparation_finished_before_reasoning',
        'original_inputs': origins, 'files': files}, ensure_ascii=False, indent=2), encoding='utf-8')
    staging.rename(destination)
    print(f'[image-preparation] snapshot saved: {destination}', flush=True)
    return destination


PREPARATION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'enum': ['READY', 'INSUFFICIENT']},
        'selected_views': {'type': 'array', 'maxItems': 8, 'uniqueItems': True,
                           'items': {'type': 'string'}},
        'findings': {'type': 'array', 'maxItems': 16, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'information_need': {'type': 'string'},
                           'source_image_ids': {'type': 'array', 'minItems': 1, 'items': {'type': 'string'}},
                           'finding': {'type': 'string'}},
            'required': ['information_need', 'source_image_ids', 'finding']}},
        'success_check': {'type': 'string', 'minLength': 1},
        'missing_information': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}},
        'residual_uncertainty': {'type': 'string'},
    },
    'required': ['status', 'selected_views', 'findings', 'success_check',
                 'missing_information', 'residual_uncertainty'],
}

PREPARATION_INSTRUCTIONS = '''IMAGE PREPARATION STAGE, before grasp/target reasoning.
Inspect the current full images and references. Identify the visual information needed to choose
a grasp AND transport destination, then edit images only to resolve those needs. Choose the sequence
yourself; there is no prescribed crop/rotation workflow and no required number of edits.
Inspect actual returned images. Continue until you judge the evidence sufficient, or until further
editing cannot resolve the specific gap within the available tool/time budget. A supported negative
finding is sufficient information; poor visibility is UNKNOWN, never proof of absence.
Check orientation, relevant garment structure, clean boundaries vs marker overlays, and the ability
to relate useful local views back to current RGB when relevant to this task. Static references show
topology only and never supply executable coordinates.
Return READY only if the next stage can choose both a grasp and a destination using these fixed
images without any further image edits. Select only useful views you have inspected, cite findings,
state the sufficiency check and any residual uncertainty. READY requires empty missing_information.
Otherwise return INSUFFICIENT with concrete missing_information; do not pretend edits succeeded.
Original images remain available downstream. Any cited derived view must be selected so it can be
delivered downstream. Do not choose the final grasp/target or output a robot action plan here.
The subsequent reasoning calls can only read these images and context, not edit or acquire images.
'''


def build_handoff(payload, sources, originals, debug_dir, output):
    """Use verified local replays, preserving ancestry and original root indices."""
    Draft202012Validator(PREPARATION_SCHEMA).validate(payload)
    if payload['status'] != 'READY':
        if not payload['missing_information']:
            raise ValueError('INSUFFICIENT requires a concrete visual information gap')
        raise ValueError('IMAGE_PREPARATION_INSUFFICIENT: '+ '; '.join(payload['missing_information']))
    if payload['missing_information'] or not payload['selected_views']:
        raise ValueError('READY requires selected views and no blocking visual gaps')
    views = {v['image_id']: v for v in sources}
    roots = {f'image_{i}' for i in range(len(originals))}
    selected = set(payload['selected_views'])
    for item in payload['findings']:
        if not set(item['source_image_ids']) <= roots | selected:
            raise ValueError('Findings cite a view not selected for the handoff')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths, aliases, lineage = list(originals), {}, {}
    for identity in payload['selected_views']:
        view = views.get(identity, {})
        if view.get('image_delivery_status') not in VERIFIED_DELIVERIES:
            raise ValueError('Selected view was not verifiably delivered to the preparation model')
        current, seen = identity, set()
        while current is not None:
            if current in seen or current not in views:
                raise ValueError('Missing or cyclic image reference chain')
            seen.add(current)
            node = views[current]
            if node.get('verification') != 'VERIFIED':
                raise ValueError('Unverified image reference chain')
            lineage[current] = copy.deepcopy(node)
            parent = node.get('parent_image_id')
            if parent is None:
                index = node.get('original_image_index')
                if type(index) is not int or not 0 <= index < len(originals) or current != f'image_{index}':
                    raise ValueError('Image ancestry does not terminate at an original input')
                with Image.open(originals[index]) as im:
                    if pixel_hash(im.convert('RGB')) != node.get('rgb_sha256'):
                        raise ValueError('Original input changed after preparation')
            current = parent
        source = Path(view['path']).resolve(strict=True)
        if Path(debug_dir).resolve() not in source.parents:
            raise ValueError('Prepared view is outside its audited invocation')
        with Image.open(source) as im:
            im = im.convert('RGB')
            if pixel_hash(im) != view.get('rgb_sha256') or list(im.size) != view['size']:
                raise ValueError('Prepared view pixels changed after verification')
            if identity not in roots:
                new_id = f'image_{len(paths)}'
                dest = output/(new_id+'.png')
                im.save(dest)
                paths.append(dest)
                aliases[new_id] = {**copy.deepcopy(view), 'image_id': new_id, 'path': str(dest.resolve())}
    id_map = {v['image_id']: v['image_id'] for v in views.values() if v['image_id'] in roots}
    derived = [i for i in payload['selected_views'] if i not in roots]
    id_map.update(zip(derived, aliases))
    findings = copy.deepcopy(payload['findings'])
    for finding in findings:
        finding['source_image_ids'] = [id_map[i] for i in finding['source_image_ids']]
    bundle = {'status': 'READY', 'findings': findings, 'success_check': payload['success_check'],
              'residual_uncertainty': payload['residual_uncertainty'],
              'prepared_view_id_map': id_map,
              'coordinate_lineage': [{k: v.get(k) for k in ('image_id', 'parent_image_id',
                  'original_image_index', 'size', 'to_parent', 'rgb_sha256')} for v in [*lineage.values(), *aliases.values()]],
              'instruction': 'Findings are model observations, not independently proven facts. '
                  'Use supplied views and original Rxxx IDs. No image editing. '
                  'Transport may cite a prepared view only when its ancestry is current clean Camera A RGB; '
                  'reference/overlay ancestry is never an executable coordinate source.'}
    (output/'handoff.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+'\n')
    return paths, bundle, list(lineage.values()), aliases


def reasoning_sources(returned_sources, lineage, aliases):
    """Restore prepared-view coordinate ancestry after re-delivery as numbered inputs."""
    result = {v['image_id']: v for v in lineage}
    for view in returned_sources:
        identity = view['image_id']
        if identity in aliases:
            original = aliases[identity]
            if view.get('verification') != 'VERIFIED' or view.get('rgb_sha256') != original['rgb_sha256']:
                raise ValueError('Prepared image changed during reasoning delivery')
            view = {**view, **{k: original.get(k) for k in ('parent_image_id', 'original_image_index', 'to_parent')}}
        result[identity] = view
    return list(result.values())
