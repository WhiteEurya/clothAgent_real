"""Host-only image IO and affine execution for registered pure skill programs."""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

from PIL import Image

from ...image_tools_mcp import ImageTools, pixel_hash
from ..common import digest, write_json
from ..policy import PolicyError
from ..skills.registry import builtin_registry


class RegisteredObservationHost:
    def __init__(self, obs, images, artifact, output, max_ops=12, *, registry=None):
        if type(max_ops) is not int or not 0 <= max_ops <= 24:
            raise PolicyError('Invalid observation operation budget')
        self.directory = Path(output)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.registry = (registry or builtin_registry()).clone()
        self.registry_hash = digest(self.registry.snapshot())
        self.skills = {s['id']: s for s in artifact['skills']}
        self.catalog, self.paths, self.input_hashes = [], [], []
        if len(obs['images']) != len(images) or not 1 <= len(images) <= 16:
            raise PolicyError('Image catalog mismatch')
        for i, (item, source) in enumerate(zip(obs['images'], images)):
            if item['image_id'] != f'image_{i}': raise PolicyError('Image IDs must match attachment order')
            target = self.directory / f'image_{i}.png'
            with Image.open(source) as im:
                if list(im.size) != item['size']: raise PolicyError('Image dimensions mismatch')
                sha = pixel_hash(im)
                if item.get('rgb_sha256') and sha != item['rgb_sha256']: raise PolicyError('Image identity mismatch')
            shutil.copyfile(source, target)
            self.input_hashes.append((Path(source), sha))
            self.catalog.append({**item, 'original_image_id': item['image_id'], 'to_original': [1,0,0,0,1,0],
                                 'rgb_sha256': sha, 'path': str(target)})
            self.paths.append(target)
        self.roots = {i['image_id']: i for i in self.catalog}
        self.tools = ImageTools(self.directory, len(images), edit_limit=max_ops)
        self.max_ops, self.ops, self.elapsed_s = max_ops, 0, 0.
        self.history, self.cache, self.seen = [], {}, set()

    def prepare(self, request, information):
        from ..information_flow import request_schema
        from ..policy import validate_schema
        validate_schema(request, request_schema(self.registry))
        if request['gap_id'] not in {i['id'] for i in information if i['status'] == 'UNKNOWN'}:
            raise PolicyError('Observation must address a currently UNKNOWN gap')
        if request['skill_id'] not in self.skills: raise PolicyError('Requested skill is not in the supplied library')
        skill = self.registry.get(request['skill_id'])
        source = self.roots.get(request['source_image_id'])
        if source is None or source['role'] != 'clean': raise PolicyError('Observation parameters must reference the current clean root')
        source_data = {k: source[k] for k in ('image_id', 'role', 'size')}
        if not skill.applicable(request, source_data): raise PolicyError('Skill input capability mismatch')
        recipe = skill.prepare(request, source_data, {'cached_views': len(self.cache), 'remaining_ops': self.max_ops-self.ops})
        paths = []
        for role in recipe['roles']:
            matches = [s for s in self.roots.values() if s['role'] == role]
            if len(matches) != 1 or matches[0]['size'] != source['size']:
                raise PolicyError('Observation requires an aligned clean/overlay pair')
            root = matches[0]
            width, height = root['size']
            chain, steps = [], []
            for operation in recipe['operations']:
                if operation['op'] == 'crop':
                    r = operation['roi']
                    box = [math.floor(r[0]*width), math.floor(r[1]*height), math.ceil(r[2]*width), math.ceil(r[3]*height)]
                    width, height = box[2]-box[0], box[3]-box[1]
                    op, args = 'crop_image', {'box': box}
                elif operation['op'] == 'rotate':
                    op, args = 'rotate_image', {'degrees_clockwise': operation['degrees']}
                    if operation['degrees'] in (90,270): width, height = height, width
                else:
                    op, args = 'resize_image', {'scale': operation['scale']}
                    width, height = max(1,round(width*operation['scale'])), max(1,round(height*operation['scale']))
                if width*height > 16_777_216 or max(width,height) > 8192 or min(width,height) < 1:
                    raise PolicyError('Output image exceeds pixel budget')
                chain.append([op,args])
                signature = digest([root['image_id'], chain])
                steps.append({'signature': signature, 'op': op, 'arguments': args})
            paths.append({'root': root, 'steps': steps})
        return {'skill': skill, 'request': request, 'recipe': recipe, 'paths': paths}

    def execute(self, requests, information):
        from ..information_flow import requests_schema
        from ..policy import validate_schema
        validate_schema(requests, requests_schema(self.registry))
        if digest(self.registry.snapshot()) != self.registry_hash: raise PolicyError('Registry changed during replay')
        for path, sha in self.input_hashes:
            with Image.open(path) as im:
                if pixel_hash(im) != sha: raise PolicyError('Source input changed')
        prepared = [self.prepare(r, information) for r in requests]
        available, finals, needed = set(self.cache), set(self.seen), 0
        for plan in prepared:
            targets = {p['steps'][-1]['signature'] for p in plan['paths']}
            if targets <= finals or (targets & finals and not plan['recipe']['reuse_existing']):
                return 'REPEATED_OBSERVATION'
            for path in plan['paths']:
                for step in path['steps']:
                    if step['signature'] not in available: needed += 1; available.add(step['signature'])
            finals |= targets
        if self.ops + needed > self.max_ops: return 'OBSERVATION_BUDGET_EXHAUSTED'
        started = time.monotonic()
        try:
            for plan in prepared:
                delivered = plan['skill'].execute(self, plan)
                plan['skill'].validate_output(self, delivered)
        finally:
            self.elapsed_s += time.monotonic()-started
            write_json(self.directory/'execution.json', {'registry': self.registry.snapshot(), 'history': self.history,
                       'catalog': self.catalog, 'host_image_ops': self.ops})
        return None

    def execute_prepared(self, plan):
        delivered, reused, lineages = [], [], []
        request = plan['request']
        for path in plan['paths']:
            root, current = path['root'], path['root']['image_id']
            for step in path['steps']:
                if step['signature'] not in self.cache:
                    view = self.tools.call(step['op'], {'image_id': current, **step['arguments']})
                    self.ops += 1
                    if not isinstance(view, dict) or 'image_id' not in view: raise PolicyError('Host image operation failed')
                    self.cache[step['signature']] = view
                else:
                    view = self.cache[step['signature']]
                    with Image.open(view['path']) as im:
                        if pixel_hash(im) != view['rgb_sha256']: raise PolicyError('Cached view changed')
                current = view['image_id']
                lineages.append({k: view.get(k) for k in ('image_id','parent_image_id','operation','arguments','to_original','to_parent','size')})
            existing = next((i for i in self.catalog if i.get('source_view_id') == current), None)
            if existing:
                item = existing; reused.append(item['image_id'])
            else:
                item = {'image_id': f'image_{len(self.paths)}', 'role': root['role'], 'original_image_id': root['image_id'],
                    'size': view['size'], 'to_original': view['to_original'], 'source_view_id': current,
                    'rgb_sha256': view['rgb_sha256'], 'gap_id': request['gap_id'], 'skill_id': request['skill_id'], 'path': view['path']}
                if len(self.paths) >= 32: raise PolicyError('Delivered view budget exhausted')
                self.catalog.append(item); self.paths.append(Path(view['path']))
            delivered.append(item)
            self.seen.add(path['steps'][-1]['signature'])
        spec = self.skills[request['skill_id']]
        self.history.append({'request': request, 'skill_version': plan['skill'].key, 'implementation_hash': plan['skill'].content_hash,
            'delivered_image_ids': [i['image_id'] for i in delivered], 'reused_image_ids': reused,
            'lineage': lineages, 'status': 'EXECUTED_NOT_YET_INTERPRETED', 'success_check': spec['success_check'],
            'on_insufficient': spec['on_insufficient']})
        return delivered

    def validate_delivered(self, delivered):
        if len(delivered) == 2 and (delivered[0]['to_original'] != delivered[1]['to_original'] or delivered[0]['size'] != delivered[1]['size']):
            raise PolicyError('Clean/overlay transform mismatch')
        for item in delivered:
            matrix = item['to_original']
            if len(matrix) != 6 or any(not math.isfinite(v) for v in matrix): raise PolicyError('Invalid affine output')
            from ..executor import inverse
            inverse(matrix)
            with Image.open(item['path']) as im:
                if list(im.size) != item['size'] or pixel_hash(im) != item['rgb_sha256']: raise PolicyError('Output identity mismatch')
        for path, sha in self.input_hashes:
            with Image.open(path) as im:
                if pixel_hash(im) != sha: raise PolicyError('Skill overwrote source input')
