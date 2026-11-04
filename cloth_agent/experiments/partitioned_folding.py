"""Graph folding with one selected arm per step. Offline proposals only.

Consumes an observed node and current-image anchors; does not recognize states,
connect robots, or assume that Cartesian proposals are reachable trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from cloth_agent.dual_arm.observation import Observation
from cloth_agent.site_observation import (
    ROOT, DEFAULT_PROFILE, load_site, convert_pose_mm_deg, rigid,
)
from .arm6_traj_replay import compile_templates

DEFAULT_GRAPH = ROOT / 'data/graphs/arm6_replay_goal_n007/graph.json'
DEFAULT_POLICY = ROOT / 'config/folding.partitioned.json'


class PartitionError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path=DEFAULT_POLICY):
    policy = json.loads(Path(path).read_text())
    if policy.get('version') != 1 or policy.get('execution_enabled') is not False:
        raise PartitionError('Expected a version 1 offline policy')
    limits = {'center_band_mm': (0., 100.), 'minimum_centerline_mm': (20., 1000.),
              'max_surface_spread_mm': (.1, 50.), 'minimum_anchor_confidence': (0., 1.),
              'approach_clearance_mm': (10., 300.), 'lift_clearance_mm': (10., 300.),
              'grasp_surface_offset_mm': (-5., 10.), 'place_surface_offset_mm': (0., 30.),
              'retreat_clearance_mm': (10., 300.)}
    for key, (low, high) in limits.items():
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not low <= value <= high:
            raise PartitionError(f'Invalid policy value: {key}')
    if policy['lift_clearance_mm'] <= policy['place_surface_offset_mm']:
        raise PartitionError('Lift must be above the release surface offset')
    return policy


def sample_pixel(observation, pixel, spread):
    if (not isinstance(pixel, list) or len(pixel) != 2
            or any(type(v) is not int for v in pixel)):
        raise PartitionError('Anchors must be integer [x, y] pixels')
    return np.asarray(observation.sample(pixel, spread), dtype=float)


def partition_frame(profile, center_a, center_b, table_normal, policy):
    """Orient the split toward physical left base; independent of pixel/collar direction."""
    normal = np.asarray(table_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        raise PartitionError('Invalid tabletop normal')
    normal = normal / np.linalg.norm(normal)
    a, b = np.asarray(center_a, dtype=float), np.asarray(center_b, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise PartitionError('Invalid centerline endpoints')
    direction = b - a
    direction -= normal * np.dot(direction, normal)
    if np.linalg.norm(direction) < policy['minimum_centerline_mm']:
        raise PartitionError('Centerline is too short on the tabletop')
    direction /= np.linalg.norm(direction)
    side = np.cross(direction, normal)
    origin = (a + b) / 2.
    bases = {arm: rigid(profile['arms'][arm]['common_from_base_m'])[:3, 3] * 1000.
             for arm in ('left', 'right')}
    if np.dot(bases['left'] - bases['right'], side) < 0:
        side = -side
    distances = {arm: float(np.dot(base - origin, side)) for arm, base in bases.items()}
    band = policy['center_band_mm']
    if distances['left'] <= band or distances['right'] >= -band:
        raise PartitionError('Centerline does not separate the physical left/right bases; verify cloth orientation')
    return dict(origin_common_mm=origin.tolist(), left_normal_common=side.tolist(),
                line_direction_common=direction.tolist(), table_normal_common=normal.tolist(),
                base_signed_distances_mm=distances, center_band_mm=band)


def signed_distance(frame, point):
    return float(np.dot(np.asarray(point) - frame['origin_common_mm'], frame['left_normal_common']))


def choose_arm(frame, grasp, center_arm=None):
    if center_arm not in (None, 'left', 'right'):
        raise PartitionError('center_arm must be left or right')
    distance = signed_distance(frame, grasp)
    band = frame['center_band_mm']
    if abs(distance) <= band:
        if center_arm is None:
            return None, 'CENTER_ARM_REQUIRED', distance
        return center_arm, 'explicit_center_arm', distance
    arm = 'left' if distance > 0 else 'right'
    if center_arm is not None:
        raise PartitionError('center_arm is only allowed inside the center band; cannot override side ownership')
    return arm, 'grasp_half', distance


def single_arm_operations(profile, arm, start, target, frame, policy):
    """Generate constant-orientation Cartesian proposals in common and arm-base frames."""
    # Use this arm's current-site orientation, never old Arm6 world-frame RPY.
    base_rpy = profile['arms'][arm]['grasp_rpy_deg']
    common_rpy = convert_pose_mm_deg(profile, [0., 0., 0., *base_rpy], arm, 'common')[3:]
    normal = np.asarray(frame['table_normal_common'])
    start, target = np.asarray(start), np.asarray(target)
    height = max(np.dot(start, normal), np.dot(target, normal)) + policy['lift_clearance_mm']
    positions = dict(
        approach=start + policy['approach_clearance_mm'] * normal,
        grasp=start + policy['grasp_surface_offset_mm'] * normal,
        lifted=start + (height - np.dot(start, normal)) * normal,
        across=target + (height - np.dot(target, normal)) * normal,
        placed=target + policy['place_surface_offset_mm'] * normal,
        retreat=target + (policy['place_surface_offset_mm'] + policy['retreat_clearance_mm']) * normal,
    )
    operations = []
    for phase, action in [('open', 'open'), ('approach', None), ('grasp', None), ('close', 'close'),
                          ('lifted', None), ('across', None), ('placed', None), ('release', 'open'), ('retreat', None)]:
        if action:
            operations.append(dict(kind='gripper', phase=phase, arm=arm, action=action))
        else:
            common = np.r_[positions[phase], common_rpy]
            operations.append(dict(kind='move_tcp', phase=phase, arm=arm,
                pose_common_mm_deg=common.tolist(),
                pose_base_mm_deg=convert_pose_mm_deg(profile, common, 'common', arm).tolist()))
    if {op['arm'] for op in operations} != {arm}:
        raise PartitionError('Every operation in a step must use the same arm')
    return operations


def plan_step(graph, templates, profile, observation, request, policy, table_normal):
    if (observation.meta.get('calibration_id') != profile['calibration_id']
            or observation.meta.get('common_frame') != profile['common_frame']
            or observation.meta.get('camera_serial') != profile['camera']['serial']):
        raise PartitionError('Observation does not match the current site')
    if (request.get('observation_id') != observation.meta['observation_id']
            or request.get('rgb_sha256') != observation.meta['sha256']['rgb.png']):
        raise PartitionError('Anchors/centerline belong to a different observation')
    confidence = request.get('confidence')
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not np.isfinite(confidence) or not policy['minimum_anchor_confidence'] <= confidence <= 1.):
        raise PartitionError('Insufficient anchor confidence')
    if request.get('anchor_source') not in ('manual', 'model', 'synthetic_demo'):
        raise PartitionError('Declare anchor_source: manual, model or synthetic_demo')
    edges = [edge for edge in graph['edges'] if edge['edge_id'] == request.get('edge_id')]
    if len(edges) != 1 or edges[0]['edge_id'] not in templates:
        raise PartitionError('Unknown or unsupported graph edge')
    edge = edges[0]
    if request.get('current_node') != edge['source_node']:
        raise PartitionError('Selected edge is not outgoing from the supplied current node')
    centerline = request.get('centerline_px')
    if not isinstance(centerline, list) or len(centerline) != 2:
        raise PartitionError('Provide two current-image centerline endpoints')
    spread = policy['max_surface_spread_mm']
    a, b = [sample_pixel(observation, point, spread) for point in centerline]
    start = sample_pixel(observation, request.get('grasp_px'), spread)
    target = sample_pixel(observation, request.get('target_px'), spread)
    frame = partition_frame(profile, a, b, table_normal, policy)
    planar = target - start
    planar -= np.asarray(frame['table_normal_common']) * np.dot(planar, frame['table_normal_common'])
    if np.linalg.norm(planar) < 10.:
        raise PartitionError('Grasp/target transfer is too short')
    arm, reason, distance = choose_arm(frame, start, request.get('center_arm'))
    target_distance = signed_distance(frame, target)
    result = dict(version=1, mode='sequential_single_gripper',
        status='PREVIEW_READY' if arm else reason, execution_allowed=False,
        robot_calls=0, model_calls=0, calibration_id=profile['calibration_id'],
        observation_id=observation.meta['observation_id'], request=request,
        current_node=edge['source_node'], expected_target_node=edge['target_node'],
        edge_id=edge['edge_id'], action_family=edge['action_family'],
        selected_arm=arm, inactive_arm=({'left':'right', 'right':'left'}[arm] if arm else None),
        selection_reason=reason, partition=frame, grasp_common_mm=start.tolist(), target_common_mm=target.tolist(),
        grasp_signed_distance_mm=distance, target_signed_distance_mm=target_distance,
        crosses_centerline=bool(distance * target_distance < 0),
        source_demonstration=dict(arm=templates[edge['edge_id']]['arm'], axis=templates[edge['edge_id']]['axis']),
        primitive='open_approach_grasp_close_lift_transfer_place_open_retreat',
        geometry_policy=policy, operations=[],
        validations=dict(current_node_visually_verified=False, centerline_visually_verified=False,
            anchor_correspondence_verified=False, ik_checked=False, collision_checked=False,
            inactive_arm_pose_checked=False, robot_start_state_checked=False),
        next_step='Observe again after execution; match an actual node before selecting the next edge')
    if arm is not None:
        result['axis'] = profile['arms'][arm]['robot_axis']
        result['operations'] = single_arm_operations(profile, arm, start, target, frame, policy)
        from cloth_agent.site_observation import base_point_mm
        points = np.array([base_point_mm(profile, point, arm) for point in (start, target)])
        # The current xArm7 site intentionally enforces only the measured
        # table-side minimum Z.  The old imported X/Y rectangle was a local
        # target sample, not a complete robot workspace and must not reject a
        # valid garment point before the active-arm IK and static-obstacle
        # proofs run.
        imported_bounds = profile['arms'][arm].get(
            'source_target_bounds_mm', profile['arms'][arm]['target_bounds_mm'])
        min_z = float(imported_bounds[2][0])
        result['within_imported_local_target_bounds'] = bool(np.all(points[:, 2] >= min_z))
        result['workspace_min_z_mm'] = min_z
        result['workspace_note'] = (
            'Current execution enforces only the measured minimum base-frame Z; '
            'X/Y and maximum Z are intentionally unrestricted here. Active-arm '
            'IK and collision proofs, including the inactive static arm model, '
            'remain authoritative.'
        )
    return result


def render_preview(observation, result, output):
    image = Image.open(observation.image).convert('RGB')
    draw = ImageDraw.Draw(image)
    request = result['request']
    a, b = map(tuple, request['centerline_px'])
    draw.line([a, b], fill='#ffe066', width=4)
    for key, color, label in [('grasp_px', '#ff5555', 'GRASP'), ('target_px', '#55eaff', 'TARGET')]:
        x, y = request[key]
        draw.ellipse((x-7,y-7,x+7,y+7), fill=color, outline='white', width=2)
        draw.text((x+10,y-18), label, fill=color, stroke_width=1, stroke_fill='black')
    draw.line([tuple(request['grasp_px']), tuple(request['target_px'])], fill='white', width=2)
    title = f"{result['edge_id']} | {result['selected_arm'] or 'CENTER: SELECT ARM'} | OFFLINE PREVIEW"
    draw.rectangle((0,0,image.width,28), fill='#182030')
    draw.text((10,8), title, fill='white')
    image.save(output)


def annotate_request(observation, graph):
    """Manual current-state/anchor input; opening this window never touches hardware."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk
    root = tk.Tk()
    root.title('双臂分工预览：每步只用一只夹爪')
    selected = tk.StringVar(value=graph['edges'][0]['edge_id'])
    center_arm = tk.StringVar(value='未指定')
    stage = tk.StringVar(value='center_a')
    confirmed = tk.BooleanVar(value=False)
    points, accepted = {}, []
    edges = {edge['edge_id']:edge for edge in graph['edges']}
    labels = {key:f"{key}: {e['source_node']} → {e['target_node']} | {e['action_family']}" for key,e in edges.items()}
    edge_label = tk.StringVar(value=labels[selected.get()])
    ttk.Label(root, text='先选符合当前衣物状态的动作，再点击中心线两端、抓点和落点。只生成离线预览。').pack(anchor='w')
    combo = ttk.Combobox(root, state='readonly', textvariable=selected, values=list(edges))
    combo.pack(anchor='w')
    combo.bind('<<ComboboxSelected>>', lambda _: edge_label.set(labels[selected.get()]))
    ttk.Label(root, textvariable=edge_label).pack(anchor='w')
    source = Image.open(observation.image).convert('RGB')
    scale = min(1., (root.winfo_screenwidth()-100)/source.width, (root.winfo_screenheight()-300)/source.height)
    size = (int(source.width*scale), int(source.height*scale))
    photo = ImageTk.PhotoImage(source.resize(size))
    canvas = tk.Canvas(root, width=size[0], height=size[1], highlightthickness=0)
    canvas.pack()
    canvas.create_image(0,0,image=photo,anchor='nw')
    bar = ttk.Frame(root); bar.pack(fill='x')
    order = ['center_a','center_b','grasp_px','target_px']
    for key, label in zip(order, ['中心线端点 1','中心线端点 2','抓取点','落点']):
        ttk.Radiobutton(bar,text=label,variable=stage,value=key).pack(side='left')
    ttk.Label(bar,text='中央抓点指定臂：').pack(side='left')
    ttk.Combobox(bar,state='readonly',textvariable=center_arm,values=['未指定','left','right'],width=8).pack(side='left')
    ttk.Checkbutton(root,text='我已核对当前状态与所选源节点，以及这一步的抓取/落点语义',variable=confirmed).pack(anchor='w')
    status = tk.StringVar(value='中心线按当前衣物位置标注；两端顺序不影响左右臂分配。')
    ttk.Label(root,textvariable=status).pack(anchor='w')
    def click(event):
        key = stage.get()
        points[key] = [min(source.width-1,int(event.x/scale)), min(source.height-1,int(event.y/scale))]
        canvas.delete('marks')
        for name,(x,y) in points.items():
            color = '#ffe066' if name.startswith('center') else '#ff5555' if name=='grasp_px' else '#55eaff'
            x,y=x*scale,y*scale
            canvas.create_oval(x-5,y-5,x+5,y+5,fill=color,tags='marks')
            canvas.create_text(x+8,y-10,text=name,fill=color,anchor='w',tags='marks')
        if 'center_a' in points and 'center_b' in points:
            canvas.create_line(*[v*scale for name in ('center_a','center_b') for v in points[name]],fill='#ffe066',width=3,tags='marks')
        stage.set(order[min(order.index(key)+1,3)])
    canvas.bind('<Button-1>',click)
    def save():
        if set(points)!=set(order) or not confirmed.get():
            status.set('请标注全部四点，并核对所选动作。'); return
        accepted.append(dict(observation_id=observation.meta['observation_id'],rgb_sha256=observation.meta['sha256']['rgb.png'],
            current_node=edges[selected.get()]['source_node'],edge_id=selected.get(),
            centerline_px=[points['center_a'],points['center_b']],grasp_px=points['grasp_px'],target_px=points['target_px'],
            confidence=1.,anchor_source='manual',operator_state_confirmed=True,
            center_arm=None if center_arm.get()=='未指定' else center_arm.get()))
        root.destroy()
    ttk.Button(root,text='生成分工预览（不执行）',command=save).pack(anchor='w')
    root.mainloop()
    if not accepted:
        raise PartitionError('Annotation cancelled')
    return accepted[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--observation', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--annotate', action='store_true', help='Manually mark current-image centerline and anchors')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.annotate and args.request:
        parser.error('Choose --request or --annotate')
    if bool(args.observation) != bool(args.request or args.annotate):
        parser.error('--observation requires --request or --annotate, and vice versa')
    profile, policy = load_site(args.profile), load_policy(args.policy)
    graph, _, templates = compile_templates(args.graph)
    if args.observation is None:
        print(json.dumps(dict(status='GRAPH_ADAPTER_OK', nodes=len(graph['nodes']), edges=len(templates),
            mode='sequential_single_gripper', supported_arms=['left','right'], execution_allowed=False,
            robot_calls=0, model_calls=0), indent=2))
        return 0
    if args.output is None or args.output.exists():
        parser.error('Provide a new --output directory')
    observation = Observation.load(args.observation)
    request = annotate_request(observation, graph) if args.annotate else json.loads(args.request.read_text())
    home = yaml.safe_load((ROOT/profile['home_source']).read_text())
    result = plan_step(graph, templates, profile, observation, request, policy, home['table_normal_common'])
    result['source_graph_sha256'] = digest(args.graph)
    result['profile_sha256'] = digest(args.profile)
    result['source_observation'] = str(args.observation.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'request.json').write_text(json.dumps(request, indent=2, ensure_ascii=False)+'\n')
    (args.output/'plan.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    render_preview(observation, result, args.output/'preview.png')
    print(json.dumps({key:result[key] for key in ('status','selected_arm','edge_id','crosses_centerline','execution_allowed','robot_calls')}, indent=2))
    return 0 if result['status']=='PREVIEW_READY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
