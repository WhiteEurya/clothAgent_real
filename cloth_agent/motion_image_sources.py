"""Resolve explicitly named RGB/view pixels on the host before depth lookup."""
from __future__ import annotations

import copy
import math

from PIL import Image

from .image_tools_mcp import pixel_hash, transform_point


def resolve_motion_sources(payload, images, verified_views=()):
    current = next(i for i, path in enumerate(images) if path.name.lower() == 'camera_a_rgb_upright.png')
    current_id = f'image_{current}'
    with Image.open(images[current]) as image:
        current_size = list(image.size)
        digest = pixel_hash(image)
    sources = {view['image_id']: view for view in verified_views}
    root = sources.get(current_id)
    if root and (root.get('rgb_sha256') != digest or root.get('verification') != 'VERIFIED'):
        raise ValueError('current RGB changed after the Claude request')
    sources[current_id] = {'image_id': current_id, 'size': current_size,
        'original_image_index': current, 'parent_image_id': None, 'verification': 'VERIFIED'}
    resolved = copy.deepcopy(payload)
    trace = []
    for index, action in enumerate(resolved.get('actions', []), 1):
        if action.get('name') != 'move':
            continue
        args = action['args']
        image_id = args.pop('image_id', None)
        if args.get('target') == 'grasp':
            if image_id is not None or args.get('pixel_xy') is not None:
                raise ValueError('fixed Rxxx grasp requires null image_id and pixel_xy')
            continue
        if args.get('target') != 'pixel' or not isinstance(image_id, str):
            raise ValueError('transport pixel requires an explicit source image_id')
        point = args.get('pixel_xy')
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError('source pixel must contain two finite numbers')
        original_point = list(point)
        selected_id = image_id
        seen = set()
        while True:
            if image_id in seen or image_id not in sources:
                raise ValueError('unknown/stale source image or incomplete transformation chain')
            seen.add(image_id)
            view = sources[image_id]
            if view.get('original_image_index') != current or view.get('verification') != 'VERIFIED':
                raise ValueError('transport point must originate in verified CURRENT Cam-A RGB, not a reference/overlay')
            w, h = view['size']
            if not (-.5 <= point[0] < w-.5 and -.5 <= point[1] < h-.5):
                raise ValueError('source pixel is outside image content or inside rotation padding')
            parent = view.get('parent_image_id')
            if parent is None:
                if image_id != current_id:
                    raise ValueError('transport point does not map to current Cam-A RGB')
                break
            point = transform_point(view['to_parent'], point)
            image_id = parent
        pixel = [int(math.floor(value + .5)) for value in point]
        if not (0 <= pixel[0] < current_size[0] and 0 <= pixel[1] < current_size[1]):
            raise ValueError('rounded current RGB pixel is outside the image')
        args['pixel_xy'] = pixel
        trace.append({'action_index': index, 'source_image_id': selected_id,
            'source_pixel_xy': original_point, 'current_image_id': current_id,
            'mapped_pixel_xy_float': list(point), 'grounding_pixel_xy': pixel,
            'rounding': 'nearest pixel center; half rounds upward'})
    return resolved, trace
