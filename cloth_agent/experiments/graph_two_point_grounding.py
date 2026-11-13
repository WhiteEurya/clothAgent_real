"""Two current pixels; reuse action-local expansion of the frozen command shape."""
import numpy as np
from scipy.spatial.transform import Rotation
from .action_local_frame import frame, expand
from .graph_minimal_grounding import POINT, validate, GroundingUncertain, edge_hash, write
from .discovery_schema import obj

SCHEMA = obj(grasp_px=POINT, target_px=POINT,
             confidence={'type': 'number', 'minimum': 0, 'maximum': 1})
PROMPT = '''Return only grasp_px, target_px and confidence. Both points are integer
[x,y] pixels in CURRENT RGB: the visible grasp fabric and receiving location for
the selected graph action. Transfer the supplied garment-relative grasp/target
relations, accounting for garment rotation without mirroring. Do not output XYZ,
yaw, intermediate waypoints, height offsets or robot code. Use null points and
confidence 0 if either correspondence is hidden, ambiguous or unsupported.
The Host transfers the stored action shape and tool orientation between these
anchors, then applies the configured contact lowering exactly once.'''


def template(edge, observation):
    ops = edge['seed_action']['exact_command']['operations']
    if (len(ops) < 2 or ops[0]['method'] != 'grasp'
            or any(o['method'] != 'move_tcp' for o in ops[1:])
            or any(set(o['arguments']) != {'right'} for o in ops)):
        raise ValueError('two-point mode requires one grasp followed by right move_tcp commands')
    poses = np.asarray([o['arguments']['right'] for o in ops], dtype=float)
    if poses.shape != (len(ops), 6) or not np.isfinite(poses).all():
        raise ValueError('invalid frozen command geometry')
    start, end = poses[0, :3]*1000, poses[-1, :3]*1000
    basis, length = frame(start, end)
    rows = []
    for i, (op, pose) in enumerate(zip(ops, poses)):
        delta = basis.T @ (pose[:3]*1000-start)
        rows.append(dict(operation_index=i, method=op['method'],
            along_fraction=float(delta[0]/length), across_fraction=float(delta[1]/length),
            height_from_grasp_mm=float(delta[2]),
            tool_rotation_relative=(basis.T @ Rotation.from_euler('xyz', pose[3:]).as_matrix()).tolist()))
    return dict(version='action-local-host-template-v2', z_policy='legacy_visual_pa',
        edge_sha256=edge_hash(edge), profile=rows, seed_horizontal_length_mm=length,
        image_size=list(observation.xyz.shape[1::-1]), camera_serial=observation.meta['camera_serial'],
        confidence_min=.8,
        semantic_limit='Two-point grounding provides no independent garment-side verification.',
        safety_limit='Stored command shape is not evidence of grasp success; current depth, workspace, IK and collision checks remain required.')


def ground(requests, graph, edge, before, folder):
    from .graph_policy_replay import grounding_reference_image
    from pathlib import Path
    folder = Path(folder)
    if graph['grasp_offset_mm'] != 0:
        raise ValueError('two-point snapshot mode requires unlowered seed grasp commands')
    compiled = template(edge, before)
    images = [('current', before.image)]
    reference = grounding_reference_image(graph, edge)
    if reference is not None:
        images.append(('action_reference', reference))
    value = requests.ask('ground_edge_two_point',
        dict(edge_id=edge['edge_id'], action_family=edge['action_family'],
             action_template=edge['action_template'], image_size=compiled['image_size']),
        images, validator=lambda v: validate(v, compiled['image_size']))
    write(folder/'two_point_anchors.json', value)
    if value['grasp_px'] is None or value['target_px'] is None or value['confidence'] < .8:
        raise GroundingUncertain('two-point anchors unresolved/low confidence')
    write(folder/'two_point_host_template.json', compiled)
    return expand(compiled, value, edge, before, folder)
